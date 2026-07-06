---
layout: post
title: "Softmax and Cross-Entropy Backward Pass"
date: 2026-07-06
author: "Shreyansh Singh"
description: "A derivation of softmax, logsoftmax, and cross-entropy backward passes, showing how the full Jacobian reduces to a row-wise dot product and why logits get the simple gradient p - y."
thumbnail: /assets/img/posts_images/softmax_cross_entropy_backward/featured.png
tags: ml math
categories: ["ML"]
giscus_comments: true
related_posts: false
permalink: "post/2026-07-06_softmax-cross-entropy-backprop/"
featured: false
toc:
  sidebar: left
pretty_table: true
---

{% include image.liquid url="/assets/img/posts_images/softmax_cross_entropy_backward/featured.png" description="Softmax backward is a structured Jacobian-vector product. Cross-entropy with logits collapses that structure to the familiar p - y gradient." width="82%" %}

Softmax followed by cross-entropy is one of those pieces of deep learning math that gets used so often that the result becomes a slogan:

$$
\frac{\partial L}{\partial z} = p - y
$$

That is correct, but it hides two details that are worth understanding.

First, softmax itself is not an elementwise function. Every output probability depends on every input logit in the same row because all classes share the same normalization denominator. Its backward pass is therefore a Jacobian-vector product, not just "take the derivative of each coordinate".

Second, the simple $$p-y$$ expression is a gradient with respect to **logits**, not probabilities. It appears only after the derivative of the log in cross-entropy cancels the coupled softmax Jacobian in exactly the right way.

This post derives both pieces: the general softmax backward pass for an arbitrary upstream gradient, and the cross-entropy/logsoftmax simplification that gives the expression used in training code.

## Setup

For one row, let the logits be

$$
z \in \mathbb{R}^K
$$

and define

$$
s = \operatorname{softmax}(z), \qquad
s_j = \frac{e^{z_j}}{\sum_{\ell=1}^{K} e^{z_\ell}}.
$$

For a batch, write the logits as

$$
Z \in \mathbb{R}^{B \times K}
$$

and the softmax outputs as

$$
S \in \mathbb{R}^{B \times K}.
$$

Softmax is applied row-wise:

$$
S_{ij} = \frac{e^{Z_{ij}}}{\sum_{\ell=1}^{K} e^{Z_{i\ell}}}.
$$

Here $$i$$ indexes the batch row. The indices $$j$$ and $$k$$ index classes, vocabulary entries, or whatever the final axis represents. Rows are independent, so we can derive everything for one row and then apply the same formula across the batch.

## The Softmax Derivative

For one batch row $$i$$, define the row-wise denominator

$$
D_i = \sum_{\ell=1}^{K} e^{Z_{i\ell}}.
$$

Then

$$
S_{ij} = \frac{e^{Z_{ij}}}{D_i}.
$$

We want the derivative

$$
\frac{\partial S_{ij}}{\partial Z_{ik}},
$$

which asks: if we nudge the logit for class $$k$$, how does the softmax probability for class $$j$$ change?

Using the quotient rule,

$$
\frac{\partial S_{ij}}{\partial Z_{ik}}
=
\frac{
  D_i \frac{\partial e^{Z_{ij}}}{\partial Z_{ik}}
  - e^{Z_{ij}} \frac{\partial D_i}{\partial Z_{ik}}
}{
  D_i^2
}.
$$

The numerator derivative depends on whether $$j=k$$:

$$
\frac{\partial e^{Z_{ij}}}{\partial Z_{ik}}
= e^{Z_{ij}}\delta_{jk},
$$

where

$$
\delta_{jk} =
\begin{cases}
1, & j=k, \\
0, & j \neq k.
\end{cases}
$$

The denominator derivative is

$$
\frac{\partial D_i}{\partial Z_{ik}}
=
\frac{\partial}{\partial Z_{ik}}
\sum_{\ell=1}^{K} e^{Z_{i\ell}}
=
e^{Z_{ik}}.
$$

Substituting both pieces gives

$$
\begin{aligned}
\frac{\partial S_{ij}}{\partial Z_{ik}}
&=
\frac{D_i e^{Z_{ij}}\delta_{jk} - e^{Z_{ij}}e^{Z_{ik}}}{D_i^2} \\
&=
\frac{e^{Z_{ij}}}{D_i}\delta_{jk}
-
\frac{e^{Z_{ij}}}{D_i}\frac{e^{Z_{ik}}}{D_i}.
\end{aligned}
$$

Since

$$
\frac{e^{Z_{ij}}}{D_i}=S_{ij}, \qquad
\frac{e^{Z_{ik}}}{D_i}=S_{ik},
$$

we get the scalar softmax derivative:

$$
\boxed{
\frac{\partial S_{ij}}{\partial Z_{ik}}
=
S_{ij}(\delta_{jk}-S_{ik})
}
$$

This compact expression contains both cases:

$$
\frac{\partial S_{ij}}{\partial Z_{ij}}
=
S_{ij}(1-S_{ij})
$$

and, for $$j \neq k$$,

$$
\frac{\partial S_{ij}}{\partial Z_{ik}}
=
-S_{ij}S_{ik}.
$$

The sign is the useful intuition. Increasing a logit's own value increases its own probability. Increasing a different class's logit increases that other class's share of the denominator, so this probability goes down.

## The Jacobian

For a fixed row $$i$$, collect all derivatives into a Jacobian matrix

$$
J_i \in \mathbb{R}^{K \times K},
\qquad
J_{i,jk} = \frac{\partial S_{ij}}{\partial Z_{ik}}.
$$

The diagonal entries are $$S_{ij}(1-S_{ij})$$. The off-diagonal entries are $$-S_{ij}S_{ik}$$. In matrix form:

$$
\boxed{
J_i = \operatorname{diag}(S_i) - S_i S_i^\top
}
$$

because

$$
{\operatorname{diag}(S_i)}_{jk} = S_{ij}\delta_{jk}
$$

and

$$
{(S_iS_i^\top)}_{jk}=S_{ij}S_{ik}.
$$

So each entry is

$$
{(\operatorname{diag}(S_i)-S_iS_i^\top)}_{jk}
=
S_{ij}\delta_{jk}-S_{ij}S_{ik}
=
S_{ij}(\delta_{jk}-S_{ik}).
$$

The important part is that the Jacobian is dense, but structured. We do not want to materialize a $$K \times K$$ matrix for every row. For a language model vocabulary, $$K$$ might be tens or hundreds of thousands. The backward pass needs the Jacobian-vector product, not the Jacobian itself.

There is also a nice reason this Jacobian is symmetric. Define

$$
\operatorname{LSE}(z) = \log \sum_{k=1}^{K} e^{z_k}.
$$

Its gradient is softmax:

$$
\nabla_z \operatorname{LSE}(z) = \operatorname{softmax}(z).
$$

So the softmax Jacobian is the Hessian of logsumexp:

$$
J_{\operatorname{softmax}}(z) = \nabla_z^2 \operatorname{LSE}(z).
$$

For a smooth scalar function, the Hessian is symmetric. That is why $$J_i^\top=J_i$$ here.

## Backpropagating Through Softmax

Let the upstream gradient for one row be

$$
G_{S_i} = \frac{\partial L}{\partial S_i}.
$$

For a vector function $$y=f(x)$$, the usual column-vector backpropagation rule is

$$
g_x = J^\top g_y.
$$

For softmax, $$J_i$$ is symmetric, so

$$
G_{Z_i} = J_i^\top G_{S_i} = J_iG_{S_i}.
$$

Now expand the multiplication:

$$
\begin{aligned}
G_{Z_i}
&=
\left(\operatorname{diag}(S_i)-S_iS_i^\top\right)G_{S_i} \\
&=
\operatorname{diag}(S_i)G_{S_i}
-
S_iS_i^\top G_{S_i}.
\end{aligned}
$$

The first term is elementwise multiplication:

$$
\operatorname{diag}(S_i)G_{S_i}
=
S_i \odot G_{S_i}.
$$

The second term has a scalar dot product inside:

$$
S_iS_i^\top G_{S_i}
=
S_i(S_i^\top G_{S_i}).
$$

Therefore,

$$
G_{Z_i}
=
S_i \odot G_{S_i}
-
S_i(S_i^\top G_{S_i}).
$$

Factoring out $$S_i$$ coordinate-wise:

$$
\boxed{
G_{Z_i}
=
S_i \odot \left(G_{S_i} - S_i^\top G_{S_i}\right)
}
$$

Elementwise:

$$
\boxed{
G_{Z,ij}
=
S_{ij}
\left(
G_{S,ij}
-
\sum_{k=1}^{K} S_{ik}G_{S,ik}
\right)
}
$$

This is the same as multiplying by the full Jacobian, but it only needs a row-wise dot product and an elementwise multiply. In PyTorch-style code:

```python
def softmax_backward(grad_out, softmax_out):
    dot = (softmax_out * grad_out).sum(dim=-1, keepdim=True)
    return softmax_out * (grad_out - dot)
```

Here `grad_out` is $$\partial L / \partial S$$ and `softmax_out` is the saved softmax output $$S$$ from the forward pass. The `keepdim=True` matters because the dot product has shape $$B \times 1$$ and should broadcast across the class dimension.

One quick sanity check: the rows of $$G_Z$$ sum to zero. This has to be true because adding the same constant to every logit in a row does not change softmax. The backward pass should not create a gradient component in that all-ones direction.

## Cross-Entropy With Logits

Now let the target be $$y$$. For a one-hot target, exactly one coordinate is $$1$$. More generally, for label smoothing or soft targets, assume $$y$$ is a normalized target distribution:

$$
\sum_{j=1}^{K} y_j = 1.
$$

The cross-entropy loss for one example is

$$
L = -\sum_{j=1}^{K} y_j \log p_j,
$$

where

$$
p = \operatorname{softmax}(z).
$$

Since

$$
\log p_j
=
\log \left(\frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}\right)
=
z_j - \log \sum_{k=1}^{K} e^{z_k},
$$

we can rewrite the loss directly as a function of logits:

$$
\begin{aligned}
L
&=
-\sum_{j=1}^{K} y_j
\left(
z_j - \log \sum_{k=1}^{K} e^{z_k}
\right) \\
&=
-\sum_{j=1}^{K} y_jz_j
+
\left(\sum_{j=1}^{K} y_j\right)
\log \sum_{k=1}^{K} e^{z_k}.
\end{aligned}
$$

Because $$\sum_j y_j=1$$,

$$
\boxed{
L = -y^\top z + \operatorname{LSE}(z)
}
$$

Differentiate with respect to logit $$z_i$$:

$$
\frac{\partial}{\partial z_i}(-y^\top z) = -y_i
$$

and

$$
\frac{\partial}{\partial z_i}\operatorname{LSE}(z)
=
\frac{e^{z_i}}{\sum_{k=1}^{K}e^{z_k}}
=
p_i.
$$

Therefore,

$$
\boxed{
\frac{\partial L}{\partial z_i} = p_i-y_i
}
$$

and in vector form,

$$
\boxed{
\frac{\partial L}{\partial z} = p-y
}
$$

For the target class $$t$$, this is

$$
\frac{\partial L}{\partial z_t} = p_t-1.
$$

For a non-target class $$i \neq t$$, it is

$$
\frac{\partial L}{\partial z_i} = p_i.
$$

So a minimal implementation for a plain batch mean loss is:

```python
probs = torch.softmax(logits, dim=-1)
B = logits.shape[0]
dlogits = probs.clone()
dlogits[torch.arange(B, device=logits.device), targets] -= 1
dlogits /= B
```

The final object `dlogits` is not a probability distribution. It is the gradient $$\partial L/\partial Z$$. For a batch mean reduction with no class weights and no ignored labels,

$$
\boxed{
\frac{\partial L_{\text{batch}}}{\partial Z}
=
\frac{1}{B}(P-Y)
}
$$

## The Same Result Through the Softmax Jacobian

The direct logsumexp derivation is the cleanest. But it is useful to see how the full softmax Jacobian disappears when chained with cross-entropy.

Starting from

$$
L = -\sum_{i=1}^{K} y_i \log p_i,
$$

the gradient with respect to probabilities is

$$
g_p = \frac{\partial L}{\partial p}
=
-\frac{y}{p}
$$

elementwise.

The softmax backward formula from above is

$$
g_z = p \odot \left(g_p - p^\top g_p\right).
$$

Compute the scalar dot product:

$$
p^\top g_p
=
\sum_{i=1}^{K} p_i\left(-\frac{y_i}{p_i}\right)
=
-\sum_{i=1}^{K} y_i
=
-1.
$$

Then

$$
\begin{aligned}
g_z
&=
p \odot \left(-\frac{y}{p} + 1\right) \\
&=
-y+p \\
&=
\boxed{p-y}.
\end{aligned}
$$

This is the key cancellation. The dense softmax Jacobian is still mathematically present, but the $$1/p_i$$ term from differentiating the log cancels it into the simple logits gradient.

## LogSoftmax Backward

Most real implementations avoid computing softmax probabilities and then taking a log separately. They use logsoftmax or a fused cross-entropy kernel, because logsumexp can be computed stably by subtracting the row maximum before exponentiating.

Define

$$
a = \operatorname{logsoftmax}(z),
\qquad
a_j = \log p_j = z_j-\operatorname{LSE}(z).
$$

Differentiate:

$$
\frac{\partial a_j}{\partial z_i}
=
\delta_{ji}
-
\frac{\partial \operatorname{LSE}(z)}{\partial z_i}
=
\delta_{ji}-p_i.
$$

Let

$$
g_a = \frac{\partial L}{\partial a}.
$$

Then

$$
\begin{aligned}
g_{z_i}
&=
\sum_{j=1}^{K} g_{a_j}
\frac{\partial a_j}{\partial z_i} \\
&=
\sum_{j=1}^{K} g_{a_j}(\delta_{ji}-p_i) \\
&=
g_{a_i} - p_i\sum_{j=1}^{K}g_{a_j}.
\end{aligned}
$$

So the vector form is

$$
\boxed{
g_z = g_a - p(\mathbf{1}^\top g_a)
}
$$

and the batch form is

$$
\boxed{
G_Z = G_A - P \odot \operatorname{rowsum}(G_A)
}
$$

where `rowsum` has shape $$B \times 1$$ and broadcasts across the final axis.

In code:

```python
def logsoftmax_backward(grad_out, softmax_out):
    rowsum = grad_out.sum(dim=-1, keepdim=True)
    return grad_out - softmax_out * rowsum
```

For negative log-likelihood or cross-entropy,

$$
L = -\sum_{j=1}^{K}y_ja_j,
$$

so

$$
g_a = -y.
$$

Because

$$
\mathbf{1}^\top g_a = -\sum_j y_j = -1,
$$

the logsoftmax backward pass gives

$$
g_z = -y - p(-1) = p-y.
$$

So all three views agree:

$$
\boxed{
\text{cross-entropy with softmax logits backward}
=
\frac{\partial L}{\partial z}
=
\operatorname{softmax}(z)-y
}
$$

and, for a plain batch mean,

$$
\boxed{
\frac{\partial L}{\partial Z}
=
\frac{1}{B}(P-Y).
}
$$

## A Small Numerical Check

Suppose the predicted probabilities and one-hot target are

$$
p = [0.659,\;0.242,\;0.099],
\qquad
y = [1,\;0,\;0].
$$

Then

$$
\frac{\partial L}{\partial z}
=
p-y
=
[-0.341,\;0.242,\;0.099].
$$

The target-class gradient is negative. Under gradient descent,

$$
z \leftarrow z - \eta \nabla_z L,
$$

subtracting a negative value increases the target logit. The non-target gradients are positive, so their logits decrease. This is exactly the behavior we want.

## What to Remember

Softmax backward is a structured Jacobian-vector product:

$$
\boxed{
G_Z = S \odot \left(G_S - \operatorname{rowsum}(S \odot G_S)\right)
}
$$

Cross-entropy with logits simplifies to:

$$
\boxed{
G_Z = P - Y
}
$$

with the appropriate scaling from the loss reduction.

The useful mental model is: softmax couples the classes through a shared denominator, but cross-entropy contributes exactly the reciprocal probability term needed to collapse that coupling. That is why training code can use a simple logits gradient even though softmax itself has a dense Jacobian.

---

&nbsp;

<script type="text/javascript" src="//downloads.mailchimp.com/js/signup-forms/popup/unique-methods/embed.js" data-dojo-config="usePlainJson: true, isDebug: false"></script>

<!-- <button style="background-color: #70ab17; color: #1770AB" id="openpopup">Subscribe to my posts!</button> -->
<div class="button_cont" align="center"><button id="openpopup" class="example_a">Subscribe to my posts!</button></div>

<style>
    .example_a {
  color: #fff !important;
  text-transform: uppercase;
  text-decoration: none;
  background: #3f51b5;
  padding: 20px;
  border-radius: 5px;
  cursor: pointer;
  display: inline-block;
  border: none;
  transition: all 0.4s ease 0s;
    }

    .example_a:hover {
  background: #434343;
  letter-spacing: 1px;
  -webkit-box-shadow: 0px 5px 40px -10px rgba(0,0,0,0.57);
  -moz-box-shadow: 0px 5px 40px -10px rgba(0,0,0,0.57);
  box-shadow: 5px 40px -10px rgba(0,0,0,0.57);
  transition: all 0.4s ease 0s;
    }
</style>

<script type="text/javascript">

function showMailingPopUp() {
    window.dojoRequire(["mojo/signup-forms/Loader"], function(L) { L.start({"baseUrl":"mc.us4.list-manage.com","uuid":"0b10ac14f50d7f4e7d11cf26a","lid":"667a1bb3da","uniqueMethods":true}) })

    document.cookie = "MCPopupClosed=;path=/;expires=Thu, 01 Jan 1970 00:00:00 UTC";
}

document.getElementById("openpopup").onclick = function() {showMailingPopUp()};

</script>

&nbsp;

<script data-name="BMC-Widget" data-cfasync="false" src="https://cdnjs.buymeacoffee.com/1.0.0/widget.prod.min.js" data-id="shreyanshsingh" data-description="Support me on Buy me a coffee!" data-message="" data-color="#FF5F5F" data-position="Right" data-x_margin="18" data-y_margin="18"></script>

Follow me on [Twitter](https://twitter.com/shreyansh_26), [Github](https://github.com/shreyansh26) or connect on [LinkedIn](https://www.linkedin.com/in/shreyansh26/).
