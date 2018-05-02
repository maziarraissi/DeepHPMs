# [Deep Hidden Physics Models](https://maziarraissi.github.io/DeepHPMs/)

A long-standing problem at the interface of artificial intelligence and applied mathematics is to devise an algorithm capable of achieving human level or even superhuman proficiency in transforming observed data into predictive mathematical models of the physical world. In the current era of abundance of data and advanced machine learning capabilities, the natural question arises: How can we automatically uncover the underlying laws of physics from high-dimensional data generated from experiments? In this work, we put forth a deep learning approach for discovering nonlinear partial differential equations from scattered and potentially noisy observations in space and time. Specifically, we approximate the unknown solution as well as the nonlinear dynamics by two deep neural networks. The first network acts as a prior on the unknown solution and essentially enables us to avoid numerical differentiations which are inherently ill-conditioned and unstable. The second network represents the nonlinear dynamics and helps us distill the mechanisms that govern the evolution of a given spatiotemporal data-set. We test the effectiveness of our approach for several benchmark problems spanning a number of scientific domains and demonstrate how the proposed framework can help us accurately learn the underlying dynamics and forecast future states of the system. In particular, we study the Burgers', Korteweg-de Vries (KdV), Kuramoto-Sivashinsky, nonlinear Schr\"{o}dinger, and Navier-Stokes equations.

For more information, please refer to the following: (https://maziarraissi.github.io/DeepHPMs/)

  - Raissi, Maziar. "[Deep Hidden Physics Models: Deep Learning of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1801.06637)." arXiv preprint arXiv:1801.06637 (2018).

## Citation

    @article{raissi2018deep,
      title={Deep Hidden Physics Models: Deep Learning of Nonlinear Partial Differential Equations},
      author={Raissi, Maziar},
      journal={arXiv preprint arXiv:1801.06637},
      year={2018}
    }
