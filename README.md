## <span style="color: yellow;">G</span><span style="color: orange;">B</span>-NeRF: NeRF Inpainting with <span style="color: yellow;">G</span>eometric Diffusion Prior and <span style="color: orange;">B</span>alanced Score

![teaser](figs/teaser.jpg)
## TODO
- [x] Release video results.
- [ ] Release the code, **we will release our code whinth 2 month!!**.



### Abstract
Recent advances in NeRF inpainting have leveraged pretrained diffusion models to enhance performance. However, these methods often yield suboptimal results due to their ineffective utilization of 2D diffusion priors. The limitations manifest in two critical aspects: the inadequate capture of geometric information by pretrained diffusion models and the suboptimal guidance provided by existing Score Distillation Sampling (SDS) methods. To address these problems, we introduce **<span style="color: yellow;">G</span><span style="color: orange;">B</span>-NeRF**, a novel framework that enhances NeRF inpainting through improved utilization of 2D diffusion priors. Our approach incorporates two key innovations: a **fine-tuning strategy** that simultaneously learns appearance and geometric priors and a specialized normal distillation loss that integrates these geometric priors into NeRF inpainting. We propose a technique called **Balanced Score Distillation (BSD)** that surpasses existing methods such as Score Distillation (SDS) and the improved version, Conditional Score Distillation (CSD). BSD offers improved inpainting quality in appearance and geometric aspects. Extensive experiments show that our method provides superior appearance fidelity and geometric consistency compared to existing approaches. Our code is coming soon!

![pipeline](figs/pipeline.jpg)
## Video Result
[![视频播放](video_result)](https://youtu.be/wnqE3VqRMMQ)