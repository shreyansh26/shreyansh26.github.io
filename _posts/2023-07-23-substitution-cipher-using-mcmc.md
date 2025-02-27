---
layout: post
title: "Solving Substitution Ciphers using Markov Chain Monte Carlo (MCMC)"
date: 2023-07-23 06:00:00
author: "Shreyansh Singh"
description: "Deciphering substitution ciphers can be framed as a Markov chain problem and a simple Monte Carlo sampling approach can help solve them very efficiently"
tags: sampling probability mcmc cryptography
categories: ["Mathematics"]
pseudocode: true
giscus_comments: true
related_posts: false
permalink: "post/2023-07-22_solving_substitution_cipher_using_mcmc/"
featured: false
toc:
  sidebar: left
---

{% include image.liquid url="/assets/img/posts_images/substitution_cipher_using_mcmc/featured.png" description="" %}

I was reading about Markov Chain Monte Carlo (MCMC) recently and discovered a very famous application of using them to decrypt substitution ciphers. This blog is meant to serve as notes on how the problem can be framed as a Markov chain and how a simple yet smart Monte Carlo sampling approach can help solve it very efficiently. In this blog post, I won't be explaining what Markov process and MCMC is, however I'll add some references for that at the end of this post.

[**Code for reference on Github**](https://github.com/shreyansh26/Solving-Substitution-Ciphers-using-MCMC)

### Assumptions

We assume a very simple setting in which we only deal with the 26 lowercase alphabets of the English language and the space character. This can be extended to more characters, but that isn't imperative to understand how the solution works. It can be extended for a larger set of characters with a bit of effort and without changing the core algorithm.

### Solution

The input we have is just the ciphertext from an English plaintext. How do we go from the ciphertext to the plaintext using MCMC?

**Scoring Function**

We need a scoring function to check how good our current state is. The state here the current mapping we have for the alphabets (for the substitution cipher).

The scoring function can be defined in multiple ways. One option can be to check fraction of decrypted words that appear in an English dictionary. Another option is to define the score to be the probability of seeing the given sequence of decrypted characters, which can be calculated as the product of consecutive bigram probabilities, using the pair-wise probabilities based on an existing long piece of English text. 

In my implementation, I have used the second option. So, if the plaintext/ciphertext is say, "good life is...", the score will be the product of probabilities - 

$$\\
\begin{align}
Pr[(g,o)] * Pr[(o,o)] * Pr[(o,d)] * Pr[(d, <space>)] * ... \textrm{and so on} 
\end{align}
\\$$

Here $$Pr[(g,o)]$$ is the estimate of the probability that $$o$$ follows $$g$$ in typical English text.

**Transition Function**

Once we have the scoring function, the transition can be defined as follows - 

**Algorithm**   

Repeat for N iterations  
$\quad$Take current mapping $m$  
$\quad$Switch the image of 2 random symbols to produce mapping $m'$  
$\quad$Compute the score for $m'$ and $m$  
$\quad$If score for $m'$ is higher than score for $m$  
$\qquad$Accept  
$\quad$If not, then with a small probability   
$\qquad$Accept   
$\quad$Else  
$\qquad$Reject


This transition function ensures that the stationary distribution of the chain puts much higher weight on better mappings.

This is how it would look when translated into code - 
<script src="https://gist.github.com/shreyansh26/9e117289ae36e1b353581672335466b9.js"></script>

The above chain runs extremely fast and solves the cipher in a couple thousand iterations. 


The entire code for this project including the bigram frequency/probability calculation and the MCMC-based solver is [here on Github](https://github.com/shreyansh26/Solving-Substitution-Ciphers-using-MCMC).

### Additional references on Markov Chains/MCMC
* [Markov Chain Monte Carlo Without all the Bullshit](https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/)
* [How would you explain Markov Chain Monte Carlo (MCMC) to a layperson?](https://stats.stackexchange.com/questions/165/how-would-you-explain-markov-chain-monte-carlo-mcmc-to-a-layperson)

------

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