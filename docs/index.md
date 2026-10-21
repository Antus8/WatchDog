---
layout: default
title: WATCHDOG
---

# WATCHDOG

## Autonomous Elderly Assistance via Attention-based Fall Detection and Trajectory Prediction

A. Longo, A. Bono, G. Guaragnella, P. Boccadoro, A. Rana, A. Petitti, T. D'Orazio

**IEEE International Conference on Robotics and Automation (ICRA) 2026**

[📄 Paper](https://ieeexplore.ieee.org/document/11696798) · [💻 Code](https://github.com/Antus8/WatchDog) · [📑 BibTeX](#citation)

<p align="center">
  <img src="./images/concept.png" alt="WATCHDOG concept" width="650">
</p>

---

## Overview

WATCHDOG is an AI-powered legged robot that watches over elderly people in cluttered home environments. It combines two attention-based models:

- a **fall-detection transformer**, which classifies short sequences of 2D body landmarks extracted with MediaPipe, and
- a **trajectory-prediction transformer**, which uses the fall model's pose embeddings to forecast where the person will move next.

A ROS package running on the robot finds the person's skeleton and 3D position with an RGB-D camera. This lets the robot detect falls and anticipate the person's motion.

---

## Method

**Fall detection.** A 3-second window sampled at 10 Hz (30 frames) is the input. Each frame holds the 2D coordinates of 11 body landmarks plus the body aspect ratio, 23 features in total. A lightweight transformer encoder pools this sequence into a fall probability.

**Trajectory prediction.** The last 0.5 s of the person's 3D positions are embedded with a GCN and encoded by a transformer encoder. A *pose–trajectory cross-attention* layer then lets the trajectory memory attend to the frozen fall transformer's pose embeddings, so the prediction is conditioned on how the body is moving. A non-autoregressive decoder with a final end-attention block predicts the next 2 s of motion in a single pass.

**On the robot.** A ROS node runs MediaPipe Holistic on the RealSense RGB stream. It deprojects the body center with the aligned depth image and publishes the person's 3D position, which forms the trajectory history fed to the model.

---

## Key features

- **Real-time fall detection** with a compact attention-based classifier
- **Trajectory prediction** conditioned on body pose, to anticipate the person's motion
- **Cluttered home environments**: runs on a legged robot with an RGB-D camera
- **ROS integration** for deployment on the real platform

---

## Resources

| Resource | Link |
|----------|------|
| 📄 **Paper** | [IEEE Xplore](https://ieeexplore.ieee.org/document/11696798) |
| 💻 **Code** | [github.com/Antus8/WatchDog](https://github.com/Antus8/WatchDog) |
| 🛠️ **Installation, data & training** | [README](https://github.com/Antus8/WatchDog#readme) |

---

## Citation

```bibtex
@inproceedings{longo2026watchdog,
  title={WATCHDOG: Autonomous Elderly Assistance via Attention-based Fall Detection and Trajectory Prediction},
  author={Longo, A and Bono, A and Guaragnella, G and Boccadoro, P and Rana, A and Petitti, A and D'Orazio, T},
  booktitle={2026 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={20685--20692},
  year={2026},
  organization={IEEE}
}
```

---

## Contact

For questions or collaborations, please open an issue on [GitHub](https://github.com/Antus8/WatchDog/issues).
