# -*- coding: utf-8 -*-
import sys
import os
import subprocess
import importlib

# --- 自动依赖检测与安装 ---
def check_and_install(package, import_name=None):
    if import_name is None:
        import_name = package
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"检测到缺失库: {import_name}，正在尝试自动安装 {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"SUCCESS: {package} 安装成功！")
        except subprocess.CalledProcessError:
            print(f"ERROR: 自动安装 {package} 失败。请手动运行: pip install {package}")
            sys.exit(1)

check_and_install("opencv-python", "cv2")
check_and_install("numpy")
check_and_install("scipy")
# ------------------------

import cv2
import numpy as np
from scipy import signal, stats
import time
import collections

class RealtimeRPPG:
    def __init__(self, video_path):
        self.video_source = video_path
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 缓冲区
        self.buffer_size = 180 
        
        # [双模态数据存储]
        # 1. 光学信号 (rPPG)
        self.signal_history = {
            'face_global': [],
            'forehead': [],
            'cheek_l': [],
            'cheek_r': [],
            'skin_global': []
        }
        self.bg_signal_history = []
        
        # 2. 运动信号 (BCG - Ballistocardiography)
        self.motion_history = [] # 存储 Y 轴位移
        
        # 光流追踪状态
        self.prev_gray = None
        self.feature_points = None
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.fps = 0
        self.min_hz = 0.75  # 45 BPM
        self.max_hz = 3.0   # 180 BPM
        
        self.roi_smooth_rect = None 
        self.smooth_factor = 0.15 
        
        self.lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        self.upper_skin = np.array([255, 173, 127], dtype=np.uint8)

    def update_roi_smooth(self, target_rect):
        if target_rect is None: return None
        target_arr = np.array(target_rect, dtype=np.float32)
        if self.roi_smooth_rect is None:
            self.roi_smooth_rect = target_arr
        else:
            self.roi_smooth_rect = (self.smooth_factor * target_arr + 
                                    (1 - self.smooth_factor) * self.roi_smooth_rect)
        return self.roi_smooth_rect.astype(int)

    def get_roi_mean_rgb(self, frame, roi_rect, use_skin_mask=True):
        x, y, w, h = roi_rect
        x, y = max(0, x), max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        if w <= 5 or h <= 5: return None
        
        roi = frame[y:y+h, x:x+w]
        
        mask = None
        if use_skin_mask:
            img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
            mask = cv2.inRange(img_ycrcb, self.lower_skin, self.upper_skin)
            if w * h > 400:
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
        
        if mask is not None:
            vals = roi[mask > 0]
        else:
            vals = roi.reshape(-1, 3)
            
        if len(vals) < 10: return None 
        
        # 简单像素清洗
        gray = vals[:, 0] * 0.114 + vals[:, 1] * 0.587 + vals[:, 2] * 0.299
        sorted_indices = np.argsort(gray)
        n_pixels = len(gray)
        # 剔除两端极值
        clean_vals = vals[sorted_indices[int(n_pixels*0.1):int(n_pixels*0.9)]]
        if len(clean_vals) == 0: return np.mean(vals, axis=0)
        return np.mean(clean_vals, axis=0)

    def process_motion_bcg(self, frame, face_rect):
        """
        [新增] BCG 运动检测模块
        使用光流法追踪面部特征点的微小垂直位移
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 如果是第一帧或特征点丢失，重新检测特征点
        if self.feature_points is None or len(self.feature_points) < 10:
            # 仅在面部区域检测特征点
            mask = np.zeros_like(gray)
            x, y, w, h = face_rect
            # 缩小区域，只取面部中心最稳定的三角区
            cv2.rectangle(mask, (x + int(w*0.3), y + int(h*0.3)), (x + int(w*0.7), y + int(h*0.8)), 255, -1)
            
            self.feature_points = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=100, qualityLevel=0.01, minDistance=5)
            self.prev_gray = gray
            self.motion_history.append(0.0) # 补位
            return
            
        # 计算光流
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.feature_points, None, **self.lk_params)
        
        if p1 is not None:
            # 筛选有效点
            good_new = p1[st==1]
            good_old = self.feature_points[st==1]
            
            if len(good_new) > 0:
                # 计算 Y 轴位移 (垂直运动)
                dy = good_new[:, 1] - good_old[:, 1]
                # 取中位数位移，抗干扰
                avg_dy = np.median(dy)
                
                # 累积位移 (积分) 得到位置曲线
                last_pos = self.motion_history[-1] if self.motion_history else 0.0
                curr_pos = last_pos + avg_dy
                self.motion_history.append(curr_pos)
                
                # 更新
                self.feature_points = good_new.reshape(-1, 1, 2)
            else:
                self.feature_points = None # 重新检测
        else:
            self.feature_points = None
            
        self.prev_gray = gray

    def process_multi_rois(self, frame, face_rect):
        """提取光信号"""
        fx, fy, fw, fh = face_rect
        visuals = [] 
        
        rois_def = {
            'face_global': (int(fx + fw*0.15), int(fy + fh*0.1), int(fw*0.7), int(fh*0.8)),
            'forehead':    (int(fx + fw*0.25), int(fy + fh*0.05), int(fw*0.5), int(fh*0.20)),
            'cheek_l':     (int(fx + fw*0.1),  int(fy + fh*0.45), int(fw*0.25), int(fh*0.25)),
            'cheek_r':     (int(fx + fw*0.65), int(fy + fh*0.45), int(fw*0.25), int(fh*0.25))
        }
        
        for name, rect in rois_def.items():
            rgb = self.get_roi_mean_rgb(frame, rect, use_skin_mask=True)
            if rgb is not None:
                self.signal_history[name].append((rgb[2], rgb[1], rgb[0]))
                visuals.append(rect)
            else:
                self._pad_history(name, self.signal_history)

        # 全画皮肤
        small = cv2.resize(frame, (320, 240))
        global_rgb = self.get_roi_mean_rgb(small, (0,0,320,240), use_skin_mask=True)
        if global_rgb is not None:
            self.signal_history['skin_global'].append((global_rgb[2], global_rgb[1], global_rgb[0]))
        else:
            self._pad_history('skin_global', self.signal_history)

        # 背景
        bg_rgb = self.get_roi_mean_rgb(frame, (frame.shape[1]-60, 10, 50, 50), use_skin_mask=False)
        if bg_rgb is not None:
            self.bg_signal_history.append((bg_rgb[2], bg_rgb[1], bg_rgb[0]))
        else:
            if self.bg_signal_history: self.bg_signal_history.append(self.bg_signal_history[-1])
            else: self.bg_signal_history.append((0,0,0))
            
        return visuals

    def _pad_history(self, name, target_dict):
        if len(target_dict[name]) > 0:
            target_dict[name].append(target_dict[name][-1])
        else:
            target_dict[name].append((0,0,0))

    # --- 核心算法 ---
    def pos_algorithm(self, r, g, b):
        r_mean = np.mean(r) + 1e-6; g_mean = np.mean(g) + 1e-6; b_mean = np.mean(b) + 1e-6
        rn, gn, bn = r/r_mean, g/g_mean, b/b_mean
        s1 = gn - bn
        s2 = gn + bn - 2 * rn
        return s1 + (np.std(s1) / (np.std(s2)+1e-6)) * s2

    def chrom_algorithm(self, r, g, b):
        r_mean = np.mean(r) + 1e-6; g_mean = np.mean(g) + 1e-6; b_mean = np.mean(b) + 1e-6
        rn, gn, bn = r/r_mean, g/g_mean, b/b_mean
        Xs = 3*rn - 2*gn
        Ys = 1.5*rn + gn - 1.5*bn
        return Xs - (np.std(Xs) / (np.std(Ys)+1e-6)) * Ys

    def calculate_metrics(self, signal_in, fps):
        """通用信号度量计算"""
        # 零相移滤波
        detrended = signal.detrend(signal_in)
        b, a = signal.butter(3, [self.min_hz/(0.5*fps), self.max_hz/(0.5*fps)], btype='band')
        
        if len(detrended) > 18: filtered = signal.filtfilt(b, a, detrended)
        else: filtered = signal.lfilter(b, a, detrended)
            
        trim = min(len(filtered)//5, 30)
        filtered = filtered[trim:] if len(filtered) > 60 else filtered
        if len(filtered) < 30: return 0, -99, 1.0, 0.0, 0.0
        
        # 频域
        nfft = max(len(filtered)*4, 2048)
        freqs, psd = signal.welch(filtered, fps, nperseg=len(filtered), nfft=nfft)
        
        mask = (freqs >= self.min_hz) & (freqs <= self.max_hz)
        valid_f = freqs[mask]
        valid_p = psd[mask]
        
        if len(valid_p) == 0: return 0, -99, 1.0, 0.0, 0.0
        
        peak_idx = np.argmax(valid_p)
        bpm = valid_f[peak_idx] * 60
        peak_freq = valid_f[peak_idx]
        
        # SNR
        bin_w, guard_w = 0.25, 0.35
        sig_mask = (valid_f >= peak_freq - bin_w) & (valid_f <= peak_freq + bin_w)
        noise_mask = (valid_f < peak_freq - guard_w) | (valid_f > peak_freq + guard_w)
        snr = 10 * np.log10(np.sum(valid_p[sig_mask]) / (np.sum(valid_p[noise_mask]) + 1e-9))
        
        # Entropy
        norm_p = valid_p / (np.sum(valid_p) + 1e-9)
        entropy = stats.entropy(norm_p) / np.log(len(valid_p))
        
        # Prominence (峰值显著性)
        peaks, props = signal.find_peaks(valid_p, prominence=0.0)
        prominence = 0.0
        if len(peaks) > 0:
            best_idx = np.argmax(props['prominences'])
            prominence = props['prominences'][best_idx] / (np.sum(valid_p) + 1e-9)
            
        return bpm, snr, entropy, prominence, filtered

    def analyze_final(self, fps):
        """终极判定：融合 Color(rPPG) 和 Motion(BCG)"""
        min_len = fps * 0.8
        
        # --- 1. 分析光信号 (rPPG) ---
        candidates = []
        for region in self.signal_history.keys():
            data = self.signal_history[region]
            if len(data) > min_len:
                rgb = np.array(data)
                # 分别跑 POS 和 CHROM
                for algo_name, sig in [('POS', self.pos_algorithm(rgb[:,0], rgb[:,1], rgb[:,2])),
                                       ('CHROM', self.chrom_algorithm(rgb[:,0], rgb[:,1], rgb[:,2]))]:
                    bpm, snr, ent, prom, wave = self.calculate_metrics(sig, fps)
                    score = snr - ent*10 + prom*20
                    candidates.append({'type':'color', 'src':f"{algo_name}-{region}", 
                                       'bpm':bpm, 'snr':snr, 'ent':ent, 'prom':prom, 'score':score})
        
        if not candidates: return False, 0, -99, 0, "无数据", "None"
        best_color = max(candidates, key=lambda x: x['score'])
        
        # --- 2. 分析运动信号 (BCG) ---
        has_motion = False
        motion_bpm, motion_snr, motion_ent = 0, -99, 1.0
        
        if len(self.motion_history) > min_len:
            # 运动信号直接处理 (它是单通道)
            motion_sig = np.array(self.motion_history)
            # 运动通常是倒相位的（头向下动是血流冲击），不过只看频率无所谓
            m_bpm, m_snr, m_ent, m_prom, _ = self.calculate_metrics(motion_sig, fps)
            motion_bpm = m_bpm
            motion_snr = m_snr
            motion_ent = m_ent
            
            # 运动信号有效性判定
            if m_snr > 0.0 and m_ent < 0.85:
                has_motion = True
        
        # --- 3. 背景信号 (环境参考) ---
        bg_rgb = np.array(self.bg_signal_history)
        bg_sig = self.pos_algorithm(bg_rgb[:,0], bg_rgb[:,1], bg_rgb[:,2])
        bg_bpm, bg_snr, _, _, _ = self.calculate_metrics(bg_sig, fps)
        
        # --- 4. 融合评分 (Bio-Score Pro) ---
        score = 0
        details = []
        
        # 因子 A: 心率合理性 (黄金区间 55-90)
        c_bpm = best_color['bpm']
        if 55 <= c_bpm <= 90: score += 40; details.append("心率完美")
        elif 50 <= c_bpm <= 100: score += 20
        else: score -= 10
        
        # 因子 B: 信号质量 (rPPG)
        c_snr = best_color['snr']
        if c_snr > 1.5: score += 20; details.append("肤色信号强")
        elif c_snr > -1.5: score += 15
        elif c_snr > -4.0: score += 5
        else: score -= 20
        
        # 因子 C: 频谱纯净度
        if best_color['prom'] > 0.3: score += 15; details.append("峰值极尖锐")
        if best_color['ent'] < 0.7: score += 15; details.append("频谱纯净")
        
        # 因子 D: 双模态一致性 (光+动) -> 最强证据
        # 如果 BCG 检测到了微动，且频率与 rPPG 一致
        if has_motion:
            diff = abs(c_bpm - motion_bpm)
            if diff < 3.0:
                score += 40 # 巨额加分
                details.append(f"光动同步(Diff={diff:.1f})")
            elif diff < 6.0:
                score += 20
                details.append("光动相关")
            else:
                # 有运动但频率不一致，说明可能是说话或随机乱动，不加分
                pass
        
        # 因子 E: 环境干扰扣分
        bg_diff = abs(c_bpm - bg_bpm)
        if bg_diff < 3.0 and bg_snr > 1.5 and c_snr < bg_snr + 3.0:
            score -= 50
            details.append("环境光干扰")
            
        # 最终裁决
        # 阈值 50 分
        is_real = score >= 50
        
        # 拦截：AI高频伪影
        if c_snr > 8.0 and best_color['ent'] > 0.9:
            is_real = False; details.append("AI高频伪影")
            
        final_reason = f"分:{score}[{','.join(details)}] 动BPM:{motion_bpm:.1f}"
        return is_real, c_bpm, c_snr, score, final_reason, best_color['src']

    def run(self):
        print(f"打开: {self.video_source}")
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened(): return
        fps = cap.get(cv2.CAP_PROP_FPS); fps = 30 if fps<=0 else fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_cnt = 0
        face_rect_raw = None
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_cnt += 1
            
            if frame_cnt % 2 == 0 or face_rect_raw is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
                if len(faces) > 0: face_rect_raw = max(faces, key=lambda f: f[2]*f[3])
            
            if face_rect_raw is not None:
                smooth_rect = self.update_roi_smooth(face_rect_raw)
                cv2.rectangle(frame, tuple(smooth_rect[:2]), (smooth_rect[0]+smooth_rect[2], smooth_rect[1]+smooth_rect[3]), (255,0,0), 1)
                
                # 1. 采集光信号
                self.process_multi_rois(frame, smooth_rect)
                
                # 2. 采集运动信号 (BCG)
                self.process_motion_bcg(frame, smooth_rect)
                
                # 可视化特征点
                if self.feature_points is not None:
                    for p in self.feature_points:
                        x, y = p.ravel()
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            cv2.putText(frame, f"Frame: {frame_cnt}/{total_frames}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.imshow('Dual-Modal Liveness', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()
        
        print("\n正在执行 双模态(rPPG + BCG) 融合分析...")
        is_real, bpm, snr, score, reason, best_src = self.analyze_final(fps)
        
        # 结果图
        final = np.zeros((400, 700, 3), dtype=np.uint8)
        color = (0, 255, 0) if is_real else (0, 0, 255)
        res_txt = "REAL PERSON" if is_real else "FAKE / AI"
        
        cv2.putText(final, res_txt, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
        cv2.putText(final, f"BPM: {bpm:.1f}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
        cv2.putText(final, f"Source: {best_src}", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
        cv2.putText(final, f"SNR: {snr:.2f} dB", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
        # 自动换行显示 reason
        y = 320
        for line in [reason[i:i+60] for i in range(0, len(reason), 60)]:
            cv2.putText(final, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            y += 25
            
        cv2.imshow('Final Result', final)
        print("="*60)
        print(f"最终判定: {'【真人】' if is_real else '【AI/伪造】'}")
        print(f"最佳来源: {best_src}")
        print(f"信噪比: {snr:.2f} dB")
        print(f"心率: {bpm:.1f} BPM")
        print(f"详细依据: {reason}")
        print("="*60)
        
        while True:
            if cv2.waitKey(100) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    input_source = 0 
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.isdigit(): input_source = int(arg)
        else: input_source = arg
    else:
        print("请输入视频路径 (或回车):")
        user_in = input(">> ").strip().strip("'").strip('"')
        if user_in:
            if user_in.isdigit(): input_source = int(user_in)
            else: input_source = user_in

    app = RealtimeRPPG(input_source)
    app.run()
