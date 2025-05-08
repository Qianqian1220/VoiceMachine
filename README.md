# VoiceMachine: Low-Latency Industrial Speech Control System

**Author**: Qianqian Bian, Qiyan Huang 

**Model Base**: 
[FireRedASR-AED](https://arxiv.org/abs/2501.14350), [CosyVoice2](https://arxiv.org/abs/2412.10117), [WeNet](https://arxiv.org/abs/2102.01547)

**Project Goal**: Real-time industrial command recognition with low latency and high noise robustness.

---

## 🔧 Overview

VoiceMachine is an automatic speech recognition (ASR) system tailored for industrial environments. It is based on FireRedASR-AED, a compact yet accurate AED-based architecture optimized for low-resource deployment. The system targets real-time command recognition in noisy factory settings.

---

## 🔍 Why FireRedASR?

- 🔹 **4× Conv Subsampling** and **Depthwise Convolution (k=33)** for fast inference  
- 🔹 Lightweight encoder (only 256 channels)  
- 🔹 Combined **CTC + AED** loss for robust alignment  
- 🔹 Strong noise robustness (SpecAug + SpecSub)  
- 🔹 RTF < 0.01 even under SNR 0dB conditions  

---

## 🧪 Fine-Tuning Strategy

We fine-tuned the FireRedASR-AED model using:

- 🔸 10.67 hours of data (AISHELL + CosyVoice2-synthesized industrial commands)  
- 🔸 Noise augmentation to simulate real factory environments  
- 🔸 Style prompting with different genders, emotions, and speaking rates

Since FireRedASR-AED lacks official fine-tuning support, WeNet was used for external fine-tuning.

---

## 📊 Results

| Metric     | Before Fine-tune | After Fine-tune (WeNet) |
|------------|------------------|--------------------------|
| CER@30dB   | 0.07%            | —                        |
| CER@0dB    | 12.85%           | ↓ 0.749% (avg)           |
| RTF        | < 0.01           | ↓ 0.65%                  |

> 🔧 Fine-tuning yields minor CER gains, but improves decoding stability and efficiency.

---

## 🚧 Limitations

- Native fine-tuning not supported in current FireRedASR-AED  
- Small-scale dataset (609 real + 50 TTS commands)  
- Possible decoding mismatches across frameworks

---

## 📌 Future Work

- Add native fine-tuning and streaming support  
- Expand to multilingual command sets  
- Deploy on Jetson Nano / Raspberry Pi for real-time edge use  
- Explore other FireRed variants (XS, S, M) for embedded scenarios

---

## 📁 Related Files & Resources

- 🔗 [Demo video](https://www.bilibili.com/video/BV1yt4y1t7Cw)  
- 🔗 [Project folder (Google Drive)](https://drive.google.com/drive/folders/1KTQJWhg_LmAbjR2zfI2R1U0tp1Fufbaj?usp=drive_link)

---

## 📄 References

- Xu et al., 2025. *FireRedASR* (arXiv:2501.14350)  
- Du et al., 2024. *CosyVoice2* (arXiv:2412.10117)  
- Yao et al., 2021. *WeNet Toolkit* (arXiv:2102.01547)  

---

## 🗨️ Questions?

Feel free to open an issue or contact the authors if you want to collaborate or discuss real-time ASR applications.
