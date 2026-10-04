# BodyPart-FB

## Background

A humanoid agent is conditioned on a latent vector $z$, which acts as a compact representation of behavioral intent. In the setup described in the project poster, proprioceptive observations from a 358-dimensional state space are passed through a backward mapping network to infer the corresponding latent representation.

BodyPart-FB studies how this forward-backward representation can preserve useful whole-body behavior while making the upper and lower body components more independent.

## Method

Let $z_{full}$, $z_{upper}$, and $z_{lower}$ denote the latent representations of a full-body motion and its upper-body and lower-body variants. Let $z_{neutral}$ be the neutral latent representation.

The additive consistency objective encourages the two body-part changes to reconstruct the full-body change:

$$
z_{full} - z_{neutral}
\approx
(z_{upper} - z_{neutral}) + (z_{lower} - z_{neutral})
$$

### Additive consistency loss

For a batch of full-body, upper-body, and lower-body latent vectors, the additive consistency loss is:

$$
L_{add} = L_{direction} + 3 L_{ortho}
$$

The direction term aligns the full-body change from the neutral latent with the sum of the upper-body and lower-body changes:

$$
L_{direction}
=
1 - \operatorname{cosine\_similarity}
\left(
z_{full} - z_{neutral},
(z_{upper} - z_{neutral}) + (z_{lower} - z_{neutral})
\right)
$$

The orthogonality term is computed per sample along the latent dimension (dim=-1), then averaged over the batch. Taking the absolute value penalizes alignment in either direction:

$$
L_{ortho}
=
\operatorname{mean}_{batch}
\left(
\left|
\operatorname{cosine\_similarity}_{dim=-1}
(z_{upper} - z_{neutral}, z_{lower} - z_{neutral})
\right|
\right)
$$

### Dimensional consistency loss

For the dimensional consistency experiment, the poster computes a separate centroid for the upper-only and lower-only latent vectors across the batch (mean(dim=0, keepdim=True)). It then sums squared deviations over the latent dimension (sum(dim=-1)) and averages over the batch:

$$
centroid_{uo} = z_{upper}.\operatorname{mean}(dim=0, keepdim=True)
$$
$$
L_1 = (z_{upper} - centroid_{uo})^2.\operatorname{sum}(dim=-1).\operatorname{mean}()
$$

$$
centroid_{lo} = z_{lower}.\operatorname{mean}(dim=0, keepdim=True)
$$
$$
L_2 = (z_{lower} - centroid_{lo})^2.\operatorname{sum}(dim=-1).\operatorname{mean}()
$$

The poster compares the original model, the additive-loss model, and the additive-plus-dimensional-loss model. It lists L1 and L2 but does not specify their aggregate weighting in the combined objective.

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
