# BodyPart-FB

## Background

A humanoid agent is conditioned on a latent vector \(z\), which acts as a compact representation of behavioral intent. In the setup described in the project poster, proprioceptive observations from a 358-dimensional state space are passed through a backward mapping network to infer the corresponding latent representation.

BodyPart-FB studies how this forward-backward representation can preserve useful whole-body behavior while making the upper and lower body components more independent.

## Method

Let \(z_{full}\), \(z_{upper}\), and \(z_{lower}\) denote the latent representations of a full-body motion and its upper-body and lower-body variants. Let \(z_{neutral}\) be the neutral latent representation.

The additive consistency objective encourages the two body-part changes to reconstruct the full-body change:

\[
z_{full} - z_{neutral}
\approx
(z_{upper} - z_{neutral}) + (z_{lower} - z_{neutral})
\]

### Additive consistency loss

The poster defines the additive consistency loss as:

\[
L_{add} = L_{direction} + 3 L_{ortho}
\]

where the direction term aligns the full-body latent change with the sum of the part-specific changes:

\[
L_{direction}
=
1 - \operatorname{cosine\_similarity}
\left(
z_{full} - z_{neutral},
(z_{upper} - z_{neutral}) + (z_{lower} - z_{neutral})
\right)
\]

The orthogonality term discourages the upper-body and lower-body latent changes from pointing in the same direction:

\[
L_{ortho}
=
\operatorname{mean}
\left(
\left|
\operatorname{cosine\_similarity}
(z_{upper} - z_{neutral}, z_{lower} - z_{neutral})
\right|
\right)
\]

### Dimensional consistency loss

The experiment also evaluates an additional dimensional consistency objective. For each body part, the poster computes a batch centroid and the mean squared distance of its latent vectors from that centroid:

\[
c_u = \operatorname{mean}(z_{upper}), \qquad
L_u = \operatorname{mean}\left(\sum (z_{upper} - c_u)^2\right)
\]

\[
c_l = \operatorname{mean}(z_{lower}), \qquad
L_l = \operatorname{mean}\left(\sum (z_{lower} - c_l)^2\right)
\]

The compared variants are the original model, the model trained with additive consistency loss, and the model trained with additive plus dimensional consistency losses.

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
