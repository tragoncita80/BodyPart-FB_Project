# BodyPart-FB

## Background

A MetaMotivo humanoid agent is conditioned on a latent vector $z$, which acts as a compact representation of behavioral intent. Proprioceptive observations from a 358-dimensional state space are passed through a backward mapping network to infer the corresponding latent representation.

The project studies how this forward-backward representation can preserve useful whole-body behavior while making the upper and lower body components more independent.

## Method

Let $z_{full}$, $z_{upper}$, and $z_{lower}$ denote the latent representations of a full-body motion and its upper-body and lower-body variants. Let $z_{neutral}$ be the neutral latent representation.

The additive consistency objective encourages the two body-part changes to reconstruct the full-body change:

$z_{full} - z_{neutral} \approx (z_{upper} - z_{neutral}) + (z_{lower} - z_{neutral})$

### Additive consistency loss

The additive consistency loss combines a direction term and an orthogonality term:

$L_{add} = L_{direction} + 3L_{ortho}$

Define each latent change relative to the neutral representation:

$\Delta z_{full}=z_{full}-z_{neutral}, \qquad \Delta z_{upper}=z_{upper}-z_{neutral}, \qquad \Delta z_{lower}=z_{lower}-z_{neutral}$

The direction and orthogonality terms use cosine similarity. For vectors a and b, define:

$c(a,b)=\frac{a\cdot b}{\sqrt{a\cdot a}\sqrt{b\cdot b}}$

The direction term aligns the full-body change with the sum of the upper-body and lower-body changes:

$L_{direction}=1-c(\Delta z_{full},\Delta z_{upper}+\Delta z_{lower})$

The orthogonality term computes the absolute cosine similarity for each sample along the latent dimension (dim=-1), then averages across a batch of size B:

$L_{ortho}=\frac{1}{B}\sum_{b=1}^{B}|c(\Delta z_{upper}^{(b)},\Delta z_{lower}^{(b)})|$

### Dimensional consistency loss

For a batch of B samples, the upper-only and lower-only latent centroids are computed across the batch:

$c_{upper}=\frac{1}{B}\sum_{b=1}^{B}z_{upper}^{(b)}, \qquad c_{lower}=\frac{1}{B}\sum_{b=1}^{B}z_{lower}^{(b)}$

The dimensional terms sum squared deviations over latent dimension d and then average across the batch:

$L_1=\frac{1}{B}\sum_{b=1}^{B}\sum_{j=1}^{d} \left(z_{upper,j}^{(b)}-c_{upper,j}\right)^2$

$L_2=\frac{1}{B}\sum_{b=1}^{B}\sum_{j=1}^{d} \left(z_{lower,j}^{(b)}-c_{lower,j}\right)^2$

## Data preparation

The project starts from the AMASS dataset, which contains 8,902 motions. Each motion is split into upper-body and lower-body versions based on SMPL skeleton indices. The full, upper-body, and lower-body versions are grouped as triplets and stored in a paired buffer for training.

## Experimental setup

- **Batch size:** 64 motion triplets
- **Compared models:** Original model, additive consistency loss, additive plus dimensional consistency loss
- **Reported measures:** Upper/lower cosine similarity, upper and lower differences, and active latent dimensions

## References

1. Andrea Tirinzoni, Ahmed Touati, Jesse Farebrother, Mateusz Guzek, Anssi Kanervisto, Yingchen Xu, Alessandro Lazaric, and Matteo Pirotta. *Zero-Shot Whole-Body Humanoid Control via Behavioral Foundation Models*.
2. Nikos Athanasiou, Mathis Petrovich, Michael J. Black, and Gül Varol. *SINC: Spatial Composition of 3D Human Motions for Simultaneous Action Generation*.

## Acknowledgement

Project advisor: Yu-Shuen Wang, Department of Computer Science, National Yang Ming Chiao Tung University.
