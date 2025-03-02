Large Language Diffusion Models  
Shen Nie  
1 *  
y  
Fengqi Zhu  
1 *  
y  
Zebin You  
1  
y  
Xiaolu Zhang  
2  
z  
Jingyang Ou  
1  
Jun Hu  
2  
z  
Jun Zhou  
2  
Yankai Lin  
1  
z  
Ji-Rong Wen  
1  
Chongxuan Li  
1  
z {  
Abstract  
Autoregressive models (ARMs) are widely re-  
garded as the cornerstone of large language mod-  
els (LLMs). We challenge this notion by intro-  
ducing  
LLaDA  
, a diffusion model trained from  
scratch under the pre-training and supervised ne-  
tuning (SFT) paradigm. LLaDA models distri-  
butions through a forward data masking process  
and a reverse process, parameterized by a vanilla  
Transformer to predict masked tokens. By op-  
timizing a likelihood bound, it provides a prin-  
cipled generative approach for probabilistic in-  
ference. Across extensive benchmarks, LLaDA  
demonstrates strong  
scalability  
, outperforming  
our self-constructed ARM baselines. Remark-  
ably, LLaDA 8B is competitive with strong L LMs  
like LLaMA3 8B in  
in-context learning  
and, af-  
ter SFT, exhibits impressive  
instruction-following  
abilities in case studies such as multi-turn dia-  
logue. Moreover, LLaDA addresses the rever-  
sal curse, surpassing GPT-4o in a reversal poem  
completion task. Our ndings establish diffu-  
sion models as a viable and promising alternative  
to ARMs, challenging the assumption that key  
LLM capabilities discussed above are inherently  
tied to ARMs. Project page and codes:  
https:  
//ml- gsai.github.io/LLaDA- demo/  
.  
1. Introduction  
What is now proved was once only imagined.  
placeholder,placeholder,placeho  
ŠWilliam Blake  
Large language models (LLMs) (  
Zhao et al.  
,  
 2023  
) fall en-  
tirely within the framework of  
generative modeling  
. Speci-  
cally, LLMs aim to capture the true but unknown language  
*  
Equal contribution  
y  
Work done during an internship at Ant  
Group  
z  
Project leaders  
1  
Gaoling School of Articial Intelli-  
gence, Renmin University of China; Beijing Key Laboratory  
of Big Data Management and Analysis Methods  
2  
Ant Group.  
{  
Correspondence to: Chongxuan Li  
<  
chongxuanli@ruc.edu.cn  
>  
.  
Preprint.  
Figure 1.  
Zero/Few-Shot Benchmarks.  
We scale LLaDA to an  
unprecedented size of 8B parameters from scratch, achieving com-  
petitive performance with strong LLMs (  
Dubey et al.  
,  
 2024  
).  
distribution  
p  
data  
(  
  
)  
by optimizing a model distribution  
p  
  
(  
  
)  
through maximum likelihood estimation, or equivalently KL  
divergence minimization between the two distributions:  
max  
  
E  
p  
data  
(  
x  
)  
log  
p  
  
(  
x  
)  
,  
min  
  
KL  
(  
p  
data  
(  
x  
)  
jj  
p  
  
(  
x  
))  
|  
{z  
}  
Generative modeling principles  
:  
(1)  
The predominant approach relies on the  
autoregressive  
mod-  
eling (ARM)Šcommonly referred to as the  
next-token pre-  
diction  
paradigmŠto dene the model distribution:  
p  
  
(  
x  
) =  
p  
  
(  
x  
1  
)  
L  
Y  
i  
=2  
p  
  
(  
x  
i  
j  
x  
1  
; : : : ; x  
i  
  
1  
)  
|  
{z  
}  
Autoregressive formulation  
;  
(2)  
where  
x  
is a sequence of length  
L  
, and  
x  
i  
is the  
i  
-th token.  
This paradigm has proven remarkably effective (  
Radford  
,  
2018  
;  
 Radford et al.  
,  
 2019  
;  
 Brown  
,  
 2020  
;  
 OpenAI  
,  
 2022  
)  
and has become the foundation of current LLMs. Despite  
its widespread adoption, a fundamental question remains  
unanswered:  
Is the autoregressive paradigm the only viable  
path to achieving the intelligence exhibited by LLMs?  
1  
arXiv:2502.09992v2  [cs.CL]  18 Feb 2025  
  
  
Large Language Diffusion Models  
We argue that the answer is  
not  
a simple ﬁyesﬂ. The key  
insight overlooked previously is: it is the  
generative mod-  
eling principles  
(i.e., Eq. (  
1  
)),  
rather than the autoregres-  
sive formulation  
(i.e., Eq. (  
2  
)) itself, that fundamentally  
underpin the essential properties of LLMs, as detailed be-  
low. However, certain inherent limitations of LLMs can be  
directly traced to their autoregressive nature.  
In particular, we argue that  
scalability  
is primarily a con-  
sequence of the interplay between Transformers (  
Vaswani  
,  
2017  
), model and data size, and  
Fisher consistency  
1  
(  
Fisher  
,  
1922  
) induced by the generative principles in Eq. (  
1  
), rather  
than a unique result of ARM. The success of diffusion trans-  
formers (  
Bao et al.  
,  
 2023  
;  
 Peebles & Xie  
,  
 2023  
) on visual  
data (  
Brooks et al.  
,  
 2024  
) supports this claim.  
Furthermore, the capabilities of  
instruction-following  
and  
in-context learning  
(  
Brown  
,  
 2020  
) appear to be intrinsic  
properties of all proper  
conditional  
generative models on  
structurally consistent linguistic tasks, rather than exclusive  
advantages of ARMs. In addition, while ARMs can be  
interpreted as a  
lossless data compressor  
(  
Deletang et al.  
;  
Huang et al.  
,  
 2024b  
), any suf ciently expressive probabilis-  
tic model can achieve similar capabilities (  
Shannon  
,  
 1948  
).  
Nevertheless, the autoregressive nature of LLMs presents  
notable challenges. For example, sequential token-by-token  
generation incurs high computational costs, and the left-  
to-right modeling limits effectiveness in reversal reasoning  
tasks (  
Berglund et al.  
,  
 2023  
). These inherent limitations  
constrain LLMs in handling longer and more complex tasks.  
Motivated by these insights, we introduce  
LLaDA  
, a  
L  
arge  
L  
anguage  
D  
iffusion with m  
A  
sking, to investigate whether  
the capabilities exhibited by LLMs can emerge from gener-  
ative modeling principles beyond ARMs in Eq. (  
2  
), thereby  
addressing the fundamental question posed earlier. In con-  
trast to traditional ARMs, LLaDA leverages a masked diffu-  
sion model (MDM) (  
Austin et al.  
,  
 2021a  
;  
 Lou et al.  
,  
 2023  
;  
Shi et al.  
,  
 2024  
;  
 Sahoo et al.  
,  
 2024  
;  
 Ou et al.  
,  
 2024  
), which  
incorporates a discrete random masking process and trains  
a  
mask predictor  
to approximate its reverse process. This  
design enables LLaDA to construct a model distribution  
with bidirectional dependencies and optimize a lower bound  
of its log-likelihood, offering an unexplored and principled  
alternative to existing LLMs.  
We adopt the standard pipeline of data preparation, pre-  
training, supervised ne-tuning (SFT), and evaluation, scal-  
ing LLaDA to an unprecedented language diffusion of size  
8B. In particular,  
LLaDA 8B  
was pre-trained from scratch  
on  
2.3 trillion tokens  
using  
0.13 million H800 GPU hours  
,  
followed by SFT on  
4.5 million pairs  
. Across diverse tasks,  
including language understanding, math, code, and Chinese,  
1  
It suggests the ability to recover the true data distribution with  
innite data, a sufciently large network and optimal training.  
LLaDA demonstrates the following contributions:  
Scalability.  
LLaDA scales effectively up to a computational  
budget of  
10  
23  
FLOPs, achieving comparable results to self-  
constructed ARM baselines trained on the same data across  
six tasks, e.g., MMLU and GSM8K.  
In-Context Learning.  
Remarkably, LLaDA 8B surpasses  
LLaMA2 7B (  
Touvron et al.  
,  
 2023  
) on nearly all 15 standard  
zero/few-shot learning tasks while performing on par with  
LLaMA3 8B (  
Dubey et al.  
,  
 2024  
).  
Instruction-Following.  
LLaDA signicantly enhances the  
ability to follow instructions after SFT, as demonstrated in  
case studies such as multi-turn dialogue.  
Reversal Reasoning.  
LLaDA effectively breaks the  
re-  
versal curse  
(  
Berglund et al.  
,  
 2023  
) with consistent per-  
formance across forward and reversal tasks. Notably, it  
outperforms GPT-4o in a reversal poem completion task.  
2. Approach  
In this section, we introduce the probabilistic formulation  
2  
,  
along with the pre-training, supervised ne-tuning, and in-  
ference procedures for LLaDA, as illustrated in Fig.  
 2  
.  
2.1. Probabilistic Formulation  
Unlike ARMs in Eq. (  
2  
), LLaDA denes a model distribu-  
tion  
p  
  
(  
x  
0  
)  
through a  
forward process  
and a  
reverse pro-  
cess  
(  
Austin et al.  
,  
 2021a  
;  
 Ou et al.  
,  
 2024  
). The forward  
process gradually masks tokens independently in  
x  
0  
until  
the sequence is fully masked at  
t  
= 1  
. For  
t  
2  
(0  
;  
1)  
, the  
sequence  
x  
t  
is partially masked, with each being masked  
with probability  
t  
or remaining unmasked with probability  
1  
  
t  
. The reverse process recovers the data distribution by  
iteratively predicting masked tokens as  
t  
moves from  
1  
to  
0  
.  
The core of LLaDA is a  
mask predictor  
, a parametric model  
p  
  
(  
j  
x  
t  
)  
that takes  
x  
t  
as input and predicts all masked tokens  
(denoted M) simultaneously. It is trained using a cross-  
entropy loss computed only on the masked tokens:  
L  
(  
  
)  
,  
  
E  
t;x  
0  
;x  
t  
"  
1  
t  
L  
X  
i  
=1  
1  
[  
x  
i  
t  
=  
M  
] log  
p  
  
(  
x  
i  
0  
j  
x  
t  
)  
#  
;  
(3)  
where  
x  
0  
is sampled from the training data,  
t  
is sampled  
uniformly from  
[0  
;  
1]  
, and  
x  
t  
is sampled from the forward  
process. The indicator function  
1  
[  
  
]  
ensures that the loss is  
computed only for masked tokens.  
Once trained, we can simulate a reverse process (see Sec.  
 2.4  
for details) parameterized by the mask predictor and dene  
the model distribution  
p  
  
(  
x  
0  
)  
as the marginal distribution  
2  
Here, we focus on the approach of LLaDA. A rigorous formu-  
lation of MDM is provided in Appendix  
 A  
 for interested readers.  
2  
  
  
Large Language Diffusion Models  
Figure 2.  
A Conceptual Overview of LLaDA.  
(a) Pre-training. LLaDA is trained on text with random masks applied independently to all  
tokens at the same ratio  
t  
˘  
U  
[0  
;  
1]  
. (b) SFT. Only response tokens are possibly masked. (c) Sampling. LLaDA simulates a diffusion  
process from  
t  
= 1  
(fully masked) to  
t  
= 0  
(unmasked), predicting all masks simultaneously at each step with exible remask strategies.  
induced at  
t  
= 0  
. Notably, the loss in Eq. (  
3  
) has been  
proven to be an upper bound on the negative log-likelihood  
of the model distribution (  
Shi et al.  
,  
 2024  
;  
 Ou et al.  
,  
 2024  
):  
  
E  
p  
data  
(  
x  
0  
)  
[log  
p  
  
(  
x  
0  
)]  
 L  
(  
  
)  
;  
(4)  
making it a principled objective for generative modeling.  
Notably, LLaDA employs a masking ratio that varies ran-  
domly between 0 and 1 while masked language models (  
De-  
vlin  
,  
 2018  
) use a xed ratio. The subtly differences have  
signicant implications, especially at scale: as shown in  
Eq. (  
4  
), LLaDA is a principled generative model with the  
potential to perform  
in-context learning  
naturally, akin to  
LLMs. Moreover, its generative perspective ensures  
Fisher  
consistency  
(  
Fisher  
,  
 1922  
) in extreme cases, suggesting  
strong  
scalability  
with large data and models.  
2.2. Pre-training  
LLaDA employs a Transformer (  
Vaswani  
,  
 2017  
) as the mask  
predictor, whose architecture is similar to existing LLMs.  
However, LLaDA does not use a causal mask, as its formu-  
lation allows it to see the entire input for predictions.  
We trained two variants of LLaDA with different sizes: 1  
billion (B) and 8B. We summarize the model architecture  
of LLaDA 8B and LLaMA3 8B (  
Dubey et al.  
,  
 2024  
) here  
and details are provided in Appendix  
 B.2  
. We have ensured  
consistency in most hyperparameters while making several  
necessary modications. We use vanilla multi-head atten-  
tion instead of grouped query attention (  
Ainslie et al.  
,  
 2023  
)  
for simplicity, as LLaDA is incompatible with KV caching,  
resulting in a different number of key and value heads. Con-  
sequently, the attention layer has more parameters, and we  
reduce the FFN dimension to maintain a comparable model  
size. Additionally, the vocabulary size differs slightly due  
to a tokenizer (  
Brown  
,  
 2020  
) adapted on our data.  
The LLaDA model is pre-trained on a dataset comprising  
2.3 trillion  
(T) tokens, adhering to a data protocol that aligns  
closely with existing large language models (LLMs) (  
Tou-  
vron et al.  
,  
 2023  
;  
 Chu et al.  
,  
 2024  
), without the incorporation  
of any special techniques. The data are derived from online  
corpora, with low-quality content ltered through manually  
designed rules and LLM-based approaches. Beyond general  
text, the dataset encompasses high-quality code, math, and  
multilingual data. The mixing of data sources and domains  
is guided by scaled-down ARMs. The pre-training process  
utilizes a xed sequence length of 4096 tokens, incurring a  
total computational cost of  
0.13 million H800 GPU hours  
,  
similar to ARMs of the same scale and dataset size.  
For a training sequence  
x  
0  
, we randomly sample  
t  
2  
[0  
;  
1]  
,  
mask each token independently with the same probability  
t  
to obtain  
x  
t  
(see Fig.  
 2  
 (a)) and estimate Eq. (  
3  
) via the  
Monte Carlo method for stochastic gradient descent training.  
In addition, following  
 Nie et al.  
 (  
2024  
), to enhance the  
ability of LLaDA to handle variable-length data, we set 1%  
of the pre-training data to a random length that is uniformly  
sampled from the range  
[1  
;  
4096]  
.  
We adopted the Warmup-Stable-Decay (  
Hu et al.  
,  
 2024  
)  
learning rate scheduler to monitor the training progress  
without interrupting continuous training. Specically, we  
linearly increased the learning rate from 0 to  
4  
  
10  
  
4  
over  
the rst 2000 iterations and maintained it at  
4  
  
10  
  
4  
. After  
processing 1.2T tokens, we decayed the learning rate to  
1  
  
10  
  
4  
and held it constant for the next 0.8T tokens to en-  
sure stable training. Finally, we linearly reduced the learning  
rate from  
1  
  
10  
  
4  
to  
1  
  
10  
  
5  
for the last 0.3T tokens. Fur-  
thermore, we utilized the AdamW optimizer (  
Loshchilov  
,  
2017  
) with a weight decay of 0.1, a batch size of 1280, and  
a local batch size of  
4  
per GPU. The 8B experiment was  
executed once, without any hyperparameter tuning.  
2.3. Supervised Fine-Tuning  
We enhance the capability of LLaDA to follow instructions  
by supervised ne-tuning (SFT) with paired data  
(  
p  
0  
; r  
0  
)  
,  
3  
  
  
Large Language Diffusion Models  
where  
p  
0  
is the prompt and  
r  
0  
denotes the response. This  
is the simplest and most basic post-training method for  
LLMs. Technically, this requires to model the  
conditional  
distribution  
p  
  
(  
r  
0  
j  
p  
0  
)  
instead of  
p  
  
(  
x  
0  
)  
in pre-training.  
The implementation is similar to pre-training. As shown in  
Fig.  
 2  
 (b), we leave the prompt unchanged and mask the  
tokens in the response independently, as done for  
x  
0  
. Then,  
we feed both the prompt and the masked response  
r  
t  
to the  
pre-trained mask predictor to compute the loss for SFT:  
  
E  
t;p  
0  
;r  
0  
;r  
t  
2  
4  
1  
t  
L  
0  
X  
i  
=1  
1  
[  
r  
i  
t  
=  
M  
] log  
p  
  
(  
r  
i  
0  
j  
p  
0  
; r  
t  
)  
3  
5  
;  
(5)  
where  
L  
0  
denotes a dynamic length specied later, and all  
other notations remain the same as before.  
Note that this approach is fully compatible with pre-training.  
Essentially, the concatenation of  
p  
0  
and  
r  
0  
can be treated  
as clean pre-training data  
x  
0  
, while the con catenation of  
p  
0  
and  
r  
t  
serves as the masked version  
x  
t  
. The process is  
identical to pre-training, with the only difference being that  
all masked tokens happen to appear in the  
r  
0  
portion.  
The LLaDA 8B model undergoes SFT on a dataset com-  
prising  
4.5 million  
pairs. Consistent with the pre-training  
process, both data preparation and training follow the SFT  
protocols utilized in existing LLMs (  
Chu et al.  
,  
 2024  
;  
 Yang  
et al.  
,  
 2024  
), without introducing any additional techniques  
to optimize LLaDA's performance. The dataset spans mul-  
tiple domains, including code, mathematics, instruction-  
following, and structured data understanding. We append  
j  
EOS  
j  
tokens to the end of short pairs in each mini-batch  
to ensure equal lengths across all data. We treat  
j  
EOS  
j  
as a  
normal token during training and remove it during sampling,  
enabling LLaDA to control the response length automati-  
cally. Please refer to Appendix  
 B.1  
 for more details.  
We train for 3 epochs on the SFT data using a similar sched-  
ule to the pre-training phase. The learning rate is linearly  
increased from 0 to  
2  
:  
5  
  
10  
  
5  
over the rst 50 iterations  
and then kept constant. During the nal  
10%  
of iterations,  
it is linearly reduced to  
2  
:  
5  
  
10  
  
6  
. Additionally, we set  
the weight decay to  
0  
:  
1  
, the global batch size to  
256  
, and  
the local batch size to  
2  
per GPU. The SFT experiment was  
executed once, without any hyperparameter tuning.  
2.4. Inference  
As a generative model, LLaDA is capable of both sampling  
new text and evaluating the likelihood of candidate text.  
We begin with the sampling. As illustrated in Fig.  
 2  
 (c),  
given a prompt  
p  
0  
, we discretize the reverse process to sam-  
ple from the model distribution  
p  
  
(  
r  
0  
j  
p  
0  
)  
, starting from a  
fully masked response. The total number of sampling steps  
is a hyperparameter, which naturally provides LLaDA with a  
trade-off between ef ciency and sample quality, as analyzed  
in Sec.  
 3.3  
. We employ uniformly distributed timesteps by  
default. In addition, the generation length is also treated as  
a hyperparameter, specifying the length of the fully masked  
sentence at the beginning of the sampling process. As de-  
tailed in Appendix  
 B.4  
, since both pre-training and SFT  
are conducted using datasets with variable lengths, the nal  
results are insensitive to this length hyperparameter.  
At an intermediate step from time  
t  
2  
(0  
;  
1]  
to  
s  
2  
[0  
; t  
)  
, we  
feed both  
p  
0  
and  
r  
t  
into the mask predictor and predict all  
masked tokens simultaneously. Subsequently, we  
remask  
s  
t  
of the predicted tokens in expectation to obtain  
r  
s  
, ensuring  
that the transition of the reverse process aligns with the  
forward process for accurate sampling (  
Austin et al.  
,  
 2021a  
).  
In principle, the remasking strategy should be purely ran-  
dom. However, inspired by the annealing tricks of sampling  
in LLMs (  
Holtzman et al.  
,  
 2019  
;  
 Brown  
,  
 2020  
), we explore  
two deterministic yet effective remasking strategies. Specif-  
ically, similarly to  
 Chang et al.  
 (  
2022  
), we remask the  
s  
t  
of predicted tokens with the lowest condence based on  
the predictions, called  
low-condence  
remasking. Addition-  
ally, for LLaDA after SFT, we can divide the sequence into  
several blocks and generate them from left to right, called  
semi-autoregressive  
remasking. Within each block, we ap-  
ply the reverse process to perform sampling. We provide  
more details and ablation studies in Appendix.  
 B.3  
.  
For conditional likelihood evaluation, we can naturally uti-  
lize the upper bound in Eq. (  
5  
). However, we nd that the  
following equivalent form (  
Ou et al.  
,  
 2024  
) exhibits lower  
variance and is more stable for evaluation:  
  
E  
l ;r  
0  
;r  
l  
"  
L  
l  
L  
X  
i  
=1  
1  
[  
r  
i  
l  
=  
M  
] log  
p  
  
(  
r  
i  
0  
j  
p  
0  
; r  
l  
)  
#  
;  
(6)  
where  
l  
is uniformly sampled from  
f  
1  
;  
2  
; : : : ; L  
g  
, and  
r  
l  
is  
obtained by uniformly sampling  
l  
tokens from  
r  
0  
without  
replacement for masking. In addition, we employ the unsu-  
pervised classier-free guidance (  
Nie et al.  
,  
 2024  
). We refer  
the readers to more details in Appendix  
 A.2  
.  
We present the training, sampling, and likelihood evaluation  
algorithms, along with theoretical details, in Appendix  
 A  
.  
3. Experiments  
We evaluate the scalability, instruction-following, and in-  
context learning capabilities of LLaDA on standard bench-  
marks, followed by analyses and case studies on more con-  
trolled datasets to provide a comprehensive assessment.  
3.1. Scalability of LLaDA on Language Tasks  
We rst investigate the  
scalability  
of LLaDA on downstream  
tasks in comparison with the ARM baselines we constructed.  
4  
  
  
Large Language Diffusion Models  
Figure 3.  
Scalability of LLaDA.  
We evaluate the performance of LLaDA and our ARM baselines trained on the same data across  
increasing computational FLOPs. LLaDA exhibits strong scalability, matching the overall performance of ARMs on six tasks.  
Specically, at the 1B scale, we ensured that LLaDA and  
ARM shared the same architecture, data, and all other cong-  
urations. At larger scales, we also report results for LLaDA  
and ARM models of slightly different sizes trained on the  
same data due to resource limit, as detailed in Appendix  
 B.2  
.  
We use the computational cost as a unied scaling metric.  
For evaluation, we focused on six standard and diverse tasks.  
As shown in Fig.  
 3  
, LLaDA demonstrates impressive scala-  
bility, with its overall trend being highly competitive with  
that of ARM. Notably, in tasks such as MMLU and GSM8K,  
LLaDA exhibits even stronger scalability. Even on tasks like  
PIQA, where performance lags, LLaDA narrows the gap  
with ARMs at larger scales. To account for the signicant  
inuence of outliers, we opted not to t quantitative scaling  
curves, avoiding potential misinterpretation. Nevertheless,  
the results clearly demonstrate the scalability of LLaDA.  
Nie et al.  
 (  
2024  
) suggests that MDM requires 16 times  
more computation than ARM to achieve the same likeli-  
hood. However, there are key differences that make the  
conclusions of this study more broadly applicable. In partic-  
ular, likelihood is a relatively indirect metric for downstream  
task performance, and diffusion optimizes a bound of the  
likelihood, making it not directly comparable to ARM. Ad-  
ditionally, we extended the scaling range from  
10  
18  
˘  
10  
20  
in  
 Nie et al.  
 (  
2024  
) to  
10  
20  
˘  
10  
23  
in this work.  
3.2. Benchmark Results  
To comprehensively evaluate the  
in-context learning  
and  
instruction-following  
capabilities of LLaDA 8B, we con-  
ducted detailed comparisons with existing LLMs (  
Touvron  
et al.  
,  
 2023  
;  
 Dubey et al.  
,  
 2024  
;  
 Chu et al.  
,  
 2024  
;  
 Yang et al.  
,  
2024  
;  
 Bi et al.  
,  
 2024  
;  
 Jiang et al.  
,  
 2023  
) of similar scale.  
The selection of tasks and evaluation protocols was aligned  
with existing studies, encompassing 15 popular benchmarks  
in general tasks, mathematics, code, and Chinese. Further  
details are provided in Appendix  
 B.5  
. For a more direct  
comparison, we re-evaluated representative LLMs (  
Touvron  
et al.  
,  
 2023  
;  
 Dubey et al.  
,  
 2024  
) in our implementation.  
As shown in Tab.  
 1  
, after pretraining on 2.3T tokens,  
LLaDA 8B demonstrates remarkable performance, surpass-  
ing LLaMA2 7B on nearly all tasks, and is overall competi-  
tive with LLaMA3 8B. LLaDA shows advantages in math  
and Chinese tasks. We conjecture that the strengths stem  
from the same factors as its relatively weaker performance  
in some tasksŠdifferences in data quality and distribution,  
largely due to the closed-source situation of LLM datasets.  
Notably, we have carefully ruled out the possibility of data  
leakage by taking GSM8K as an example. First, as shown  
in Fig.  
 3  
, LLaDA outperformed ARM baselines regarding  
GSM8K. Moreover, the conclusion remains on a fully un-  
seen GSM8K-like task (  
Ye et al.  
,  
 2024  
) in Appendix  
 B.7  
.  
Further, Tab.  
 2  
 compares the performance of LLaDA 8B In-  
struct with existing LLMs. We observed that SFT improved  
LLaDA's performance on most downstream tasks. A few  
metrics, such as MMLU, showed declines, and we conjec-  
ture may be due to the suboptimal quality of the SFT data.  
Overall, since we did not perform alignment with reinforce-  
ment learning (RL), our results are slightly behind LLaMA3  
8B Instruct, though the gaps in many metrics remain small.  
Notably, even with only SFT, LLaDA demonstrates impres-  
5  
  
  
Large Language Diffusion Models  
Table 1.  
Benchmark Results of Pre-trained LLMs.  
  
indicates that LLaDA 8B Base, LLaMA2 7B Base, and LLaMA3 8B Base are  
evaluated under the same protocol, detailed in Appendix  
 B.5  
. Results indicated by  
y  
and  
{  
are sourced from  
 Chu et al.  
 (  
2024  
);  
 Yang et al.  
(  
2024  
) and  
 Bi et al.  
 (  
2024  
) respectively. The numbers in parentheses represent the number of shots used for evaluation. ﬁ-ﬂ indicates  
unknown data.  
LLaDA 8B  
  
LLaMA3 8B  
  
LLaMA2 7B  
  
Qwen2 7B  
y  
Qwen2.5 7B  
y  
Mistral 7B  
y  
Deepseek 7B  
{  
Model  
Diffusion AR AR  
AR AR AR AR  
Training tokens  
2.3T 15T 2T  
7T 18T - 2T  
General Tasks  
MMLU  
65.9  
(5) 65.4 (5) 45.9 (5)  
70.3 (5) 74.2 (5) 64.2 (5) 48.2 (5)  
BBH  
49.8 (3)  
57.6  
(3) 37.3 (3)  
62.3 (3) 70.4 (3) 56.1 (3) 39.5 (3)  
ARC-C  
47.9 (0)  
53.1  
(0) 46.3 (0)  
60.6 (25) 63.7 (25) 60.0 (25) 48.1 (0)  
Hellaswag  
72.5 (0)  
79.1  
(0) 76.0 (0)  
80.7 (10) 80.2 (10) 83.3 (10) 75.4 (0)  
TruthfulQA  
46.4  
(0) 44.0 (0) 39.0 (0)  
54.2 (0) 56.4 (0) 42.2 (0) -  
WinoGrande  
74.8 (5)  
77.3  
(5) 72.5 (5)  
77.0 (5) 75.9 (5) 78.4 (5) 70.5 (0)  
PIQA  
74.4 (0)  
80.6  
(0) 79.1 (0)  
- - - 79.2 (0)  
Mathematics & Science  
GSM8K  
70.7  
(4) 53.1 (4) 14.3 (4)  
80.2 (4) 85.4 (4) 36.2 (4) 17.4 (8)  
Math  
27.3  
(4) 15.1 (4) 3.2 (4)  
43.5 (4) 49.8 (4) 10.2 (4) 6.0 (4)  
GPQA  
26.1  
(5) 25.9 (5) 25.7 (5)  
30.8 (5) 36.4 (5) 24.7 (5) -  
Code  
HumanEval  
33.5 (0)  
34.2  
(0) 12.8 (0)  
51.2 (0) 57.9 (0) 29.3 (0) 26.2 (0)  
HumanEval-FIM  
73.8  
(2) 73.3 (2) 26.9 (2)  
- - - -  
MBPP  
38.2 (4)  
47.4  
(4) 18.4 (4)  
64.2 (0) 74.9 (0) 51.1 (0) 39.0 (3)  
Chinese  
CMMLU  
69.9  
(5) 50.7 (5) 32.5 (5)  
83.9 (5) - - 47.2 (5)  
C-Eval  
70.5  
(5) 51.7 (5) 34.0 (5)  
83.2 (5) - - 45.0 (5)  
Table 2.  
Benchmark Results of Post-trained LLMs.  
LLaDA only employs an SFT procedure while other models have extra reinforcement  
learning (RL) alignment.  
  
indicates that LLaDA 8B Instruct, LLaMA2 7B Instruct, and LLaMA3 8B Instruct are evaluated under the  
same protocol, detailed in Appendix  
 B.5  
. Results indicated by  
y  
and  
{  
are sourced from  
 Yang et al.  
 (  
2024  
) and  
 Bi et al.  
 (  
2024  
) respectively.  
The numbers in parentheses represent the number of shots used for in-context learning. ﬁ-ﬂ indicates unknown data.  
LLaDA 8B  
  
LLaMA3 8B  
  
LLaMA2 7B  
  
Qwen2 7B  
y  
Qwen2.5 7B  
y  
Gemma2 9B  
y  
Deepseek 7B  
{  
Model  
Diffusion AR AR  
AR AR AR AR  
Training tokens  
2.3T 15T 2T  
7T 18T 8T 2T  
Post-training  
SFT SFT+RL SFT+RL  
SFT+RL SFT+RL SFT+RL SFT+RL  
Alignment pairs  
4.5M - -  
0.5M + - 1M + 0.15M - 1.5M + -  
General Tasks  
MMLU  
65.5 (5)  
68.4  
(5) 44.1 (5)  
- - - 49.4 (0)  
MMLU-pro  
37.0 (0)  
41.9  
(0) 4.6 (0)  
44.1 (5) 56.3 (5) 52.1 (5) -  
Hellaswag  
74.6 (0)  
75.5  
(0) 51.5 (0)  
- - - 68.5 (-)  
ARC-C  
88.5  
(0) 82.4 (0) 57.3 (0)  
- - - 49.4 (-)  
Mathematics & Science  
GSM8K  
78.6  
(4) 78.3 (4) 29.0 (4)  
85.7 (0) 91.6 (0) 76.7 (0) 63.0 (0)  
Math  
26.6 (0)  
29.6  
(0) 3.8 (0)  
52.9 (0) 75.5 (0) 44.3 (0) 15.8 (0)  
GPQA  
31.8 (5)  
31.9  
(5) 28.4 (5)  
34.3 (0) 36.4 (0) 32.8 (0) -  
Code  
HumanEval  
47.6 (0)  
59.8  
(0) 16.5 (0)  
79.9 (0) 84.8 (0) 68.9 (0) 48.2 (-)  
MBPP  
34.2 (4)  
57.6  
(4) 20.6 (4)  
67.2 (0) 79.2 (0) 74.9 (0) 35.2 (-)  
6  
  
  
Large Language Diffusion Models  
Table 3.  
Comparison in the Poem Completion Task  
.  
Forward Reversal  
GPT-4o (2024-08-06)  
82.7  
34.3  
Qwen2.5 7B Instruct 75.9 3 8.0  
LLaDA 8B Instruct 48.8  
42.4  
sive instruction-following abilities, as detailed in Sec.  
 3.4  
.  
We leave RL-based alignment for future work.  
Overall, despite the lack of data transparency, we have made  
every effort to adopt standardized procedures and introduce  
diverse tasks, we believe they sufciently demonstrate the  
extraordinary capabilities of LLaDA, which is the only com-  
petitive non-autoregressive model to our knowledge.  
3.3. Reversal Reasoning and Analyses  
To quantify the reversal reasoning ability of mod-  
els (  
Berglund et al.  
,  
 2023  
), we follow the protocol estab-  
lished in  
 Allen-Zhu & Li  
 (  
2023  
). Specically, we construct  
a dataset of 496 famous Chinese poem sentence pairs. Given  
a sentence from a poem, models are tasked with generating  
the subsequent line (forward) or the preceding line (rever-  
sal) without additional ne-tuning. Examples can be found  
in Appendix  
 B.8  
. This setting provides a straightforward  
and more realistic evaluation compared to previous stud-  
ies (  
Nie et al.  
,  
 2024  
;  
 Kitouni et al.  
,  
 2024  
).  
As shown in Tab.  
 3  
, LLaDA effectively addresses the  
rever-  
sal curse  
(  
Berglund et al.  
,  
 2023  
), demonstrating consistent  
zero-shot performance across both forward and reversal  
tasks. In contrast, both Qwen 2.5 and GPT-4o exhibit a  
signicant gap between the two. The results on forward  
generation conrm that both ARMs are strong, beneting  
from signicantly larger datasets and greater computational  
resources than LLaDA. However, LLaDA outperforms both  
by a large margin in the reversal task.  
We emphasize that we did not design anything special for  
reversal tasks. Intuitively, LLaDA treats tokens uniformly  
without inductive bias, leading to balanced performance.  
See more details in Appendix  
 A.2  
.  
We also analyze the effect of remasking strategies and sam-  
pling steps, as detailed in Appendix  
 B.3  
 and Appendix  
 B.6  
.  
3.4. Case Studies  
We present samples generated by LLaDA 8B Instruct in  
Tab.  
 4  
, showcasing its instruction-following capabilities.  
First, the table illustrates LLaDA's ability to generate co-  
herent, uent, and extended text in a non-autoregressive  
manner. Second, it highlights the model's multi-turn dia-  
logue capability, effectively retaining conversation history  
and producing contextually appropriate responses across  
multiple languages. Such  
chat  
capabilities of LLaDA are  
impressive, as it departs from conventional ARMs for the  
rst time, to the best of our knowledge. See more case  
studies on remasking and reasoning tasks in Appendix  
 B.9  
.  
4. Related Work  
Diffusion models (  
Sohl-Dickstein et al.  
,  
 2015  
;  
 Ho et al.  
,  
2020  
;  
 Song et al.  
,  
 2020  
) have excelled in visual domains but  
remain unveried for LLMs despite extensive efforts.  
A simple approach is to continuousize text data and apply  
diffusion models directly (  
Li et al.  
,  
 2022  
;  
 Gong et al.  
,  
 2022  
;  
Han et al.  
,  
 2022  
;  
 Strudel et al.  
,  
 2022  
;  
 Chen et al.  
,  
 2022  
;  
Dieleman et al.  
,  
 2022  
;  
 Richemond et al.  
,  
 2022  
;  
 Wu et al.  
,  
2023  
;  
 Mahabadi et al.  
,  
 2024  
;  
 Ye et al.  
,  
 2023b  
). Alternatively,  
some methods model continuous parameters of discrete dis-  
tributions instead (  
Lou & Ermon  
,  
 2023  
;  
 Graves et al.  
,  
 2023  
;  
Lin et al.  
,  
 2023  
;  
 Xue et al.  
,  
 2024  
). However, scalability  
remains a challenge, as a 1B parameter model requires  
64  
times the compute  
of an ARM to achieve comparable per-  
formance (  
Gulrajani & Hashimoto  
,  
 2024  
).  
Another approach replaces continuous diffusion with dis-  
crete processes featuring new forward and reverse dy-  
namics (  
Austin et al.  
,  
 2021a  
), leading to numerous vari-  
ants (  
Hoogeboom et al.  
,  
 2021b  
;  
a  
;  
 He et al.  
,  
 2022  
;  
 Campbell  
et al.  
,  
 2022  
;  
 Meng et al.  
,  
 2022  
;  
 Reid et al.  
,  
 2022  
;  
 Sun et al.  
,  
2022  
;  
 Kitouni et al.  
,  
 2023  
;  
 Zheng et al.  
,  
 2023  
;  
 Chen et al.  
,  
2023  
;  
 Ye et al.  
,  
 2023a  
;  
 Gat et al.  
,  
 2024  
;  
 Zheng et al.  
,  
 2024  
;  
Sahoo et al.  
,  
 2024  
;  
 Shi et al.  
,  
 2024  
). Notably,  
 Lou et al.  
(  
2023  
) showed that masked diffusion, as a special case of  
discrete diffusion, achieves perplexity comparable to or sur-  
passing ARMs at GPT-2 scale.  
 Ou et al.  
 (  
2024  
) established  
fundamental theoretical results, which motivated our model  
design, training, and inference (see Appendix  
 A  
).  
 Nie et al.  
(  
2024  
) explored how MDM can be leveraged for language  
tasks such as question answering at GPT-2 scale.  
 Gong et al.  
(  
2024  
) ne-tune ARMs in the MDM formulation. However,  
improvements are conned to certain metrics, and it remains  
unclear whether this approach can yield a foundation model  
comparable to strong LLMs under a comprehensive evalua-  
tion.  
In comparison, this study scales MDM to an unprecedented  
size of 8B parameters from scratch, achieving performance  
comparable to leading LLMs such as LLaMA 3.  
Additionally, a parallel line of work on image genera-  
tion (  
Chang et al.  
,  
 2022  
;  
 2023  
) aligns well with the appli-  
cation of MDMs to text data. Moreover, MDMs have also  
shown promise in domains such as protein generation (  
Wang  
et al.  
,  
 2024b  
;  
c  
), where they have achieved promising results.  
Notably,  
 Kou et al.  
 (  
2024  
);  
 Xu et al.  
 (  
2025  
) demonstrate the  
potential of using distillation to accelerate MDMs sampling,  
7  
  
  
Large Language Diffusion Models  
Table 4.  
Visualization of the Sampling Process and a Generated Multi-round Dialogue.  
In the response of LLaDA, darker colors  
indicate tokens predicted in the later stages of sampling, while lighter colors correspond to earlier predictions.  
Sampling Process  
User  
Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour.  
How many kilometers can she run in 8 hours?  
LLaDA  
Multi-round Dialogue Case  
User  
Could you please share with me what the initial two lines of the renowned poem `The  
Road Not Taken'?  
LLaDA  
Certainly! The initial two lines of the renowned poem ﬁThe Road Not Takenﬂ by Robert  
Frost are: ﬁTwo roads diverged in a yellow wood, And sorry I could not travel both.ﬂ These  
lines set the stage for the poem, introducing the idea of a speaker facing a choice between  
two paths.  
User Please help me translate into Chinese.  
LLaDA Sure! The rst two lines of ﬁThe Road Not Takenﬂ by Robert Frost can be translated into  
Chinese as: ﬁ  
$  
a  
ï  
  
ﬂ  
(  
Ä  
r  
—  
  
Š  
-  
  
W  
¾  
  
  
ý  
  
ö  
p  
ﬂ  
User Please translate into German.  
LLaDA Sure! The rst two lines of ﬁThe Road Not Takenﬂ by Robert Frost can be translated into  
German as: ﬁZwei Wege trennten sich im gelben Wald, und leider konnte ich nicht beide  
tun.ﬂ  
User  
Please also write a poem about life choices. I require 5 sentences, and the rst word of  
each sentence starts with C as the rst letter.  
LLaDA  
Certainly! Here's a poem about life choices, with each sentence starting with the letter ﬁCﬂ:  
Choices cascade through life's journey, Creating connections with each decision, Careful  
consideration leads to better paths, Courageous choices shape our destiny, Choosing wisely  
guides us through the years.  
further enhancing their efciency.  
5. Conclusion and Discussion  
In the middle of difculty lies opportunity.  
placeholder,placeholder,placeh  
ŠAlbert Einstein  
We introduce  
LLaDA  
, a principled and previously unex-  
plored approach to large language modeling based on dif-  
fusion models. LLaDA demonstrates strong capabilities in  
scalability, in-context learning, and instruction-following,  
achieving performance comparable to strong LLMs. In addi-  
tion, LLaDA off ers unique advantages such as bidirectional  
modeling and enhanced robustness, effectively addressing  
several inherent limitations of existing LLMs. Our ndings  
not only  
establish diffusion models as a viable and promis-  
ing alternative  
but also  
challenge the prevailing assumption  
that these essential capabilities are inherently tied to ARMs  
.  
While promising, the full potential of diffusion models re-  
mains to be fully explored. Several limitations of this work  
present signicant opportunities for future research.  
Due to computational constraints, direct comparisons be-  
tween LLaDA and ARMsŠsuch as training on identical  
datasetsŠwere restricted to a computational budget of less  
than  
10  
23  
FLOPs. To allocate resources for training the  
largest possible LLaDA model and showcasing its poten-  
tial, we were unable to scale the ARM baseline to the same  
extent. Moreover, no specialized attention mechanisms  
or position embeddings were designed for LLaDA, nor  
were any system-level architectural optimizations applied.  
On the inference side, our exploration of guidance mecha-  
nisms (  
Dhariwal & Nichol  
,  
 2021  
;  
 Ho & Salimans  
,  
 2022  
) re-  
mains preliminary, and LLaDA currently exhibits sensitivity  
to inference hyperparameters. Furthermore, LLaDA has yet  
to undergo alignment with reinforcement learning (  
Ouyang  
et al.  
,  
 2022  
;  
 Rafailov et al.  
,  
 2024  
), which is crucial for im-  
proving its performance and alignment with human intent.  
8  
  
  
Large Language Diffusion Models  
Looking forward, the scale of LLaDA is still smaller than  
leading counterparts (  
Achiam et al.  
,  
 2023  
;  
 Dubey et al.  
,  
2024  
;  
 Google  
,  
 2024  
;  
 Anthropic  
,  
 2024  
;  
 Yang et al.  
,  
 2024  
;  
Liu et al.  
,  
 2024  
), highlighting the need for further scaling  
to fully assess its capabilities. In addition, LLaDA's abil-  
ity to handle multi-modal data remains unexplored. The  
impact of LLaDA on prompt tuning techniques (  
Wei et al.  
,  
2022  
) and its integration into agent-based systems (  
Park  
et al.  
,  
 2023  
;  
 Wang et al.  
,  
 2024a  
) is yet to be fully under-  
stood. Finally, a systematic investigation into post-training  
for LLaDA could facilitate the development of O1-like sys-  
tems (  
OpenAI  
,  
 2024  
;  
 Guo et al.  
,  
 2025  
).  
Impact Statement  
This work challenges the prevailing dominance of autore-  
gressive modeling in LLMs by proposing a diffusion-based  
alternative. Our ndings open new avenues for exploring  
alternative probabilistic paradigms in natural language pro-  
cessing, with potential applications in conversational AI,  
code generation, and complex reasoning tasks.  
However, diffusion-based models, like traditional LLMs,  
raise similar societal concerns. These include the environ-  
mental impact of large-scale training, the potential misuse  
for generating harmful or misleading content, and the ampli-  
cation of biases present in training data. Addressing these  
challenges is critical to ensuring the responsible develop-  
ment and deployment of diffusion-based language models.  
Acknowledgements  
This work was supported by the National Natural Science  
Foundation of China (No. 92470118); Beijing Natural Sci-  
ence Foundation (No. L247030); Beijing Nova Program  
(No. 20220484044); Major Innovation & Planning Interdis-  
ciplinary Platform for the ﬁDouble-First Classﬂ Initiative,  
Renmin University of China; the Fundamental Research  
Funds for the Central Universities, the Research Funds of  
Renmin University of China (22XNKJ13); Ant Group Re-  
search Fund. The work was partially done at the Engineering  
Research Center of Next-Generation Intelligent Search and  
Recommendation, Ministry of Education.  
References  
Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I.,  
Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S.,  
Anadkat, S., et al. Gpt-4 technical report.  
arXiv preprint  
arXiv:2303.08774  
, 2023.  
Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y.,  
Lebron, F., and Sanghai, S. Gqa: Training generalized  
multi-query transformer models from multi-head check-  
points. In  
Proceedings of the 2023 Conference on Em-  
pirical Methods in Natural Language Processing  
, pp.  
4895Œ4901, 2023.  
Allen-Zhu, Z. and Li, Y. Physics of Language Mod-  
els: Part 3.2, Knowledge Manipulation.  
ArXiv e-prints  
,  
abs/2309.14402, September 2023. Full version available  
at  
http://arxiv.org/abs/2309.14402  
.  
Anonymous. Interpolating autoregressive and discrete de-  
noising diffusion language models. In  
The Thirteenth  
International Conference on Learning Representations  
,  
2025. URL  
https://openreview.net/forum?  
id=tyEyYT267x  
.  
Anthropic. Claude 3.5 sonnet, 2024. URL  
https://www.  
anthropic.com/news/claude- 3- 5- sonnet  
.  
Austin, J., Johnson, D. D., Ho, J., Tarlow, D., and Van  
Den Berg, R. Structured denoising diffusion models in  
discrete state-spaces.  
Advances in Neural Information  
Processing Systems  
, 34:17981Œ17993, 2021a.  
Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski,  
H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al.  
Program synthesis with large language models.  
arXiv  
preprint arXiv:2108.07732  
, 2021b.  
Bao, F., Nie, S., Xue, K., Cao, Y., Li, C., Su, H., and Zhu,  
J. All are worth words: A vit backbo ne for diffusion  
models. In  
Proceedings of the IEEE/CVF Conference  
on Computer Vision and Pattern Recognition  
, pp. 22669Œ  
22679, 2023.  
Bavarian, M., Jun, H., Tezak, N., Schulman, J., McLeavey,  
C., Tworek, J., and Chen, M. Efcient training of  
language models to ll in the middle.  
arXiv preprint  
arXiv:2207.14255  
, 2022.  
Berglund, L., Tong, M., Kaufmann, M., Balesni, M., Stick-  
land, A. C., Korbak, T., and Evans, O. The reversal curse:  
Llms trained onﬂ a is bﬂ fail to learnﬂ b is aﬂ.  
arXiv  
preprint arXiv:2309.12288  
, 2023.  
Bi, X., Chen, D., Chen, G., Chen, S., Dai, D., Deng, C.,  
Ding, H., Dong, K., Du, Q., Fu, Z., et al. Deepseek llm:  
Scaling open-source language models with longtermism.  
arXiv preprint arXiv:2401.02954  
, 2024.  
Bisk, Y., Zellers, R., Gao, J., Choi, Y., et al. Piqa: Reasoning  
about physical commonsense in natural language. In  
Pro-  
ceedings of the AAAI conference on articial intelligence  
,  
2020.  
Brooks, T., Peebles, B., Holmes, C., DePue, W.,  
Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman,  
T., Luhman, E., Ng, C., Wang, R., and Ramesh,  
A. Video generation models as world simulators.  
2024. URL  
https://openai.com/research/  
video- generation- models- as- world- simulators  
.  
9  
  
  
Large Language Diffusion Models  
Brown, T. B. Language mo dels are few-shot learners.  
arXiv  
preprint arXiv:2005.14165  
, 2020.  
Campbell, A., Benton, J., De Bortoli, V., Rainforth, T., Deli-  
giannidis, G., and Doucet, A. A continuous time frame-  
work for discrete denoising models.  
Advances in Neural  
Information Processing Systems  
, 35:28266Œ28279, 2022.  
Chang, H., Zhang, H., Jiang, L., Liu, C., and Freeman,  
W. T. Maskgit: Masked generative image transformer. In  
Proceedings of the IEEE/CVF Conference on Computer  
Vision and Pattern Recognition  
, pp. 11315Œ11325, 2022.  
Chang, H., Zhang, H., Barber, J., Maschinot, A., Lezama, J.,  
Jiang, L., Yang, M.-H., Murphy, K., Freeman, W. T.,  
Rubinstein, M., et al. Muse: Text-to-image genera-  
tion via masked generative transformers.  
arXiv preprint  
arXiv:2301.00704  
, 2023.  
Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. D. O.,  
Kaplan, J. , Edwards, H., Burda, Y., Joseph, N., Brockman,  
G., et al. Evaluating large language models trained on  
code.  
arXiv preprint arXiv:2107.03374  
, 2021.  
Chen, T., Zhang, R., and Hinton, G. Analog bits: Gen-  
erating discrete data using diffusion models with self-  
conditioning.  
arXiv preprint arXiv:2208.04202  
, 2022.  
Chen, Z., Yuan, H., Li, Y., Kou, Y., Zhang, J., and Gu, Q.  
Fast sampling via de-randomization for discrete diff usion  
models.  
arXiv preprint arXiv:2312.09193  
, 2023.  
Chu, Y., Xu, J., Yang, Q., Wei, H., Wei, X., Guo, Z., Leng,  
Y., Lv, Y., He, J., Lin, J., et al. Qwen2-audio technical  
report.  
arXiv preprint arXiv:2407.10759  
, 2024.  
Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A.,  
Schoenick, C., and Tafjord, O. Think you have solved  
question answering? try arc, the ai2 reasoning challenge.  
arXiv preprint arXiv:1803.05457  
, 2018.  
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H.,  
Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano,  
R., et al. Training veriers to solve math word problems.  
arXiv preprint arXiv:2110.14168  
, 2021.  
Deletang, G., Ruoss, A., Duquenne, P.-A., Catt, E., Ge-  
newein, T., Mattern, C., Grau-Moya, J., Wenliang, L. K.,  
Aitchison, M., Orseau, L., et al. Language modeling is  
compression. In  
The Twelfth International Conference on  
Learning Representations  
.  
Devlin, J. Bert: Pre-training of deep bidirectional trans-  
formers for language understanding.  
arXiv preprint  
arXiv:1810.04805  
, 2018.  
Dhariwal, P. and Nichol, A. Diffusion models beat gans  
on image synthesis.  
Advances in neural information  
processing systems  
, 34:8780Œ8794, 2021.  
Dieleman, S., Sartran, L., Roshannai, A., Savinov, N.,  
Ganin, Y., Richemond, P. H., Doucet, A., Strudel, R.,  
Dyer, C., Durkan, C., et al. Continuous diffusion for  
categorical data.  
arXiv preprint arXiv:2211.15089  
, 2022.  
Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle,  
A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan,  
A., et al. The llama 3 herd of models.  
arXiv preprint  
arXiv:2407.21783  
, 2024.  
Fisher, R. A. On the mathematical foun dations of theoretical  
statistics.  
Philosophical transactions of the Royal Society  
of London. Series A, containing papers of a mathematical  
or physical character  
, 222(594-604):309Œ368, 1922.  
Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPo,  
A., Foster, C., Golding, L., Hsu, J., Le Noac'h, A., Li,  
H., McDonell, K., Muennighoff, N., Ociepa, C., Phang,  
J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika,  
L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou,  
A. A framework for few-shot language model evaluation,  
07 2024. URL  
https://zenodo.org/records/  
12608602  
.  
Gat, I., Remez, T., Shaul, N., Kreuk, F., Chen, R. T., Syn-  
naeve, G., Adi, Y., and Lipman, Y. Discrete ow match-  
ing.  
arXiv preprint arXiv:2407.15595  
, 2024.  
Gong, S., Li, M., Feng, J., Wu, Z., and Kong, L. Diffuseq:  
Sequence to sequence text generation with diff usion mod-  
els.  
arXiv preprint arXiv:2210.08933  
, 2022.  
Gong, S., Agarwal, S., Zhang, Y., Ye, J., Zheng, L., Li, M.,  
An, C., Zhao, P., Bi, W., Han, J., et al. Scaling diffu-  
sion language models via adaptation from autoregressive  
models.  
arXiv preprint arXiv:2410.17891  
, 2024.  
Google. Our next-generation model: Gemini 1.5, 2024.  
URL  
https://blog.google/technology/ai/  
google- gemini- next- generation- model- february- 2024  
.  
Graves, A., Srivastava, R. K., Atkinson, T., and  
Gomez, F. Bayesian ow networks.  
arXiv preprint  
arXiv:2308.07037  
, 2023.  
Gulrajani, I. and Hashimoto, T. B. Likelihood-based diffu-  
sion language models.  
Advances in Neural Information  
Processing Systems  
, 36, 2024.  
Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R.,  
Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: In-  
centivizing reasoning capability in llms via reinforcement  
learning.  
arXiv preprint arXiv:2501.12948  
, 2025.  
Han, X., Kumar, S., and Tsvetkov, Y. Ssd-lm: Semi-  
autoregressive simplex-based diffusion language model  
for text generation and modular control.  
arXiv preprint  
arXiv:2210.17432  
, 2022.  
10  
  
  
Large Language Diffusion Models  
He, Z., Sun, T., Wang, K., Huang, X., and Qiu, X. Diffu-  
sionbert: Improving generative masked language models  
with diffusion models.  
arXiv preprint arXiv:2211.15029  
,  
2022.  
Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika,  
M., Song, D., and Steinhardt, J. Measuring mas-  
sive multitask language understanding.  
arXiv preprint  
arXiv:2009.03300  
, 2020.  
Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart,  
S., Tang, E., Song, D., and Steinhardt, J. Measuring math-  
ematical problem solving with the math dataset.  
arXiv  
preprint arXiv:2103.03874  
, 2021.  
Ho, J. and Salimans, T. Classier-free diffusion guidance.  
arXiv preprint arXiv:2207.12598  
, 2022.  
Ho, J., Jain, A., and Abbeel, P. Denoising diffusion proba-  
bilistic models.  
Advances in neural information process-  
ing systems  
, 33:6840Œ6851, 2020.  
Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E.,  
Cai, T., Rutherford, E., Casas, D. d. L., Hendricks, L. A.,  
Welbl, J., Clark, A., et al. Training compute-optimal  
large language models.  
arXiv preprint arXiv:2203.15556  
,  
2022.  
Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. The  
curious case of neural text degeneration.  
arXiv preprint  
arXiv:1904.09751  
, 2019.  
Hoogeboom, E., Gritsenko, A. A., Bastings, J., Poole, B.,  
Berg, R. v. d., and Salimans, T. Autoregressive diff usion  
models.  
arXiv preprint arXiv:2110.02037  
, 2021a.  
Hoogeboom, E., Nielsen, D., Jaini, P., Forr  
´  
e, P., and Welling,  
M. Argmax ows and multinomial diffusion: Learning  
categorical distributions.  
Advances in Neural Information  
Processing Systems  
, 34:12454Œ12465, 2021b.  
Hu, S., Tu, Y., Han, X., He, C., Cui, G., Long, X., Zheng, Z.,  
Fang, Y., Huang, Y., Zhao, W., et al. Minicpm: Unveiling  
the potential of small language models with scalable train-  
ing strategies.  
arXiv preprint arXiv:2404.06395  
, 2024.  
Huang, Y., Bai, Y., Zhu, Z., Zhang, J., Zhang, J., Su, T., Liu,  
J., Lv, C., Zhang, Y., Fu, Y., et al. C-eval: A multi-level  
multi-discipline chinese evaluation suite for foundation  
models.  
Advances in Neural Information Processing  
Systems  
, 36, 2024a.  
Huang, Y., Zhang, J., Shan, Z., and He, J. Compres-  
sion represents intelligence linearly.  
arXiv preprint  
arXiv:2404.09937  
, 2024b.  
Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C.,  
Chaplot, D. S., Casas, D. d. l., Bressand, F., Lengyel, G.,  
Lample, G., Saulnier, L., et al. Mistral 7b.  
arXiv preprint  
arXiv:2310.06825  
, 2023.  
Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B.,  
Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and  
Amodei, D. Scaling laws for neural language models.  
arXiv preprint arXiv:2001.08361  
, 2020.  
Kitouni, O., Nolte, N., Hensman, J., and Mitra, B. Disk: A  
diff usion model for structured knowledge.  
arXiv preprint  
arXiv:2312.05253  
, 2023.  
Kitouni, O., Nolte, N., Bouchacourt, D., Williams, A., Rab-  
bat, M., and Ibrahim, M. The factorization curse: Which  
tokens you predict underlie the reversal curse and more.  
arXiv preprint arXiv:2406.05183  
, 2024.  
Kou, S., Hu, L., He, Z., Deng, Z., and Zhang, H. Cllms:  
Consistency large language models.  
arXiv preprint  
arXiv:2403.00835  
, 2024.  
Li, H., Zhang, Y., Koto, F., Yang, Y., Zhao, H., Gong, Y.,  
Duan, N., and Baldwin, T. Cmmlu: Measuring mas-  
sive multitask language understanding in chinese.  
arXiv  
preprint arXiv:2306.09212  
, 2023.  
Li, X., Thickstun, J., Gulrajani, I., Liang, P. S., and  
Hashimoto, T. B. Diffusion-lm improves controllable  
text generation.  
Advances in Neural Information Process-  
ing Systems  
, 35:4328Œ4343, 2022.  
Lin, S., Hilton, J., and Evans, O. Truthfulqa: Measuring  
how models mimic human falsehoods.  
arXiv preprint  
arXiv:2109.07958  
, 2021.  
Lin, Z., Gong, Y., Shen, Y., Wu, T., Fan, Z., Lin, C., Duan,  
N., and Chen, W. Text generation with diffusion language  
models: A pre-training approach with continuous para-  
graph denoise. In  
International Conference on Machine  
Learning  
, pp. 21051Œ21064. PMLR, 2023.  
Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao,  
C., Deng, C., Zhang, C., Ruan, C., et al. Deepseek-v3  
technical report.  
arXiv preprint arXiv:2412.19437  
, 2024.  
Loshchilov, I. Decoupled weight decay regularization.  
arXiv  
preprint arXiv:1711.05101  
, 2017.  
Lou, A. and Ermon, S. Reected diffusion models, 2023.  
Lou, A., Meng, C., and Ermon, S. Discrete diffusion lan-  
guage modeling by estimating the ratios of the data distri-  
bution.  
arXiv preprint arXiv:2310.16834  
, 2023.  
Mahabadi, R. K., Ivison, H., Tae, J., Henderson, J., Beltagy,  
I., Peters, M. E., and Cohan, A. Tess: Text-to-text self-  
conditioned simplex diffusion, 2024.  
11  
  
  
Large Language Diffusion Models  
Meng, C., Choi, K., Song, J., and Ermon, S. Concrete score  
matching: Generalized score matching for discrete data.  
Advances in Neural Information Processing Systems  
, 35:  
34532Œ34545, 2022.  
Nie, S., Zhu, F., Du, C., Pang, T., Liu, Q., Zeng, G., Lin, M.,  
and Li, C. Scaling up masked diffusion models on text.  
arXiv preprint arXiv:2410.18514  
, 2024.  
OpenAI. ChatGPT: Optimizing Language Models for Di-  
alogue.  
OpenAI blog  
, November 2022. URL  
https:  
//openai.com/blog/chatgpt/  
.  
OpenAI. Learning to reason with llms, 2024.  
URL  
https://openai.com/index/  
learning- to- reason- with- llms/  
.  
Ou, J., Nie, S., Xue, K., Zhu, F., Sun, J., Li, Z., and Li,  
C. Your absorbing discrete diffusion secretly models the  
conditional distributions of clean data.  
arXiv preprint  
arXiv:2406.03736  
, 2024.  
Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C.,  
Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A.,  
et al. Training language models to follow instructions  
with human feedback.  
Advances in neural information  
processing systems  
, 35:27730Œ27744, 2022.  
Park, J. S., O'Brien, J., Cai, C. J., Morris, M. R., Liang,  
P., and Bernstein, M. S. Generative agents: Interactive  
simulacra of human behavior. In  
Proceedings of the 36th  
annual acm symposium on user interface software and  
technology  
, pp. 1Œ22, 2023.  
Peebles, W. and Xie, S. Scalable diffusion models with  
transformers. In  
Proceedings of the IEEE/CVF Interna-  
tional Conference on Computer Vision  
, pp. 4195Œ4205,  
2023.  
Radford, A. Improving language understanding by genera-  
tive pre-training, 2018.  
Radford, A., Wu, J., Child, R., Luan, D., Amodei, D.,  
Sutskever, I., et al. Language models are unsupervised  
multitask learners.  
OpenAI blog  
, 1(8):9, 2019.  
Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Er-  
mon, S., and Finn, C. Direct preference optimization:  
Your language model is secretly a reward model.  
Ad-  
vances in Neural Information Processing Systems  
, 36,  
2024.  
Reid, M., Hellendoorn, V. J., and Neubig, G. Diffuser:  
Discrete diffusion via edit-based reconstruction, 2022.  
Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang,  
R. Y., Dirani, J., Michael, J., and Bowman, S. R. Gpqa:  
A graduate-level google-proo f q&a benchmark.  
arXiv  
preprint arXiv:2311.12022  
, 2023.  
Richemond, P. H., Dieleman, S., and Doucet, A. Categorical  
sdes with simplex diffusion, 2022.  
Sahoo, S. S., Arriola, M., Schiff, Y., Gokaslan, A., Marro-  
quin, E., Chiu, J. T., Rush, A., and Kuleshov, V. Simple  
and effective masked diffusion language models.  
arXiv  
preprint arXiv:2406.07524  
, 2024.  
Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y.  
Winogrande: An adversarial winograd schema challenge  
at scale.  
Communications of the ACM  
, 64(9):99Œ106,  
2021.  
Shannon, C. E. A mathematical theory of communication.  
The Bell system technical journal  
, 27(3):379Œ423, 1948.  
Shazeer, N. Glu variants improve transformer.  
arXiv  
preprint arXiv:2002.05202  
, 2020.  
Shi, J., Han, K., Wang, Z., Doucet, A., and Titsias, M. K.  
Simplied and generalized masked diffusion for discrete  
data.  
arXiv preprint arXiv:2406.04329  
, 2024.  
Shih, A., Sadigh, D., and Ermon, S. Training and infer-  
ence on any-order autoregressive models the right way.  
In  
Proceedings of the 31th International Conference on  
Machine Learning  
, 2022.  
Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and  
Ganguli, S. Deep unsupervised learning using nonequi-  
librium thermodynamics. In  
International conference on  
machine learning  
, pp. 2256Œ2265. PMLR, 2015.  
Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Er-  
mon, S., and Poole, B. Score-based generative modeling  
through stochastic differential equations.  
arXiv preprint  
arXiv:2011.13456  
, 2020.  
Strudel, R., Tallec, C., Altch  
´  
e, F., Du, Y., Ganin, Y., Men-  
sch, A., Grathwohl, W., Savinov, N., Dieleman, S., Sifre,  
L., et al. Self-conditioned embedding diffusion for text  
generation.  
arXiv preprint arXiv:2211.04236  
, 2022.  
Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y.  
Roformer: Enhanced transformer with rotary position  
embedding.  
Neurocomputing  
, 568:127063, 2024.  
Sun, H., Yu, L., Dai, B., Schuurmans, D., and Dai, H. Score-  
based continuous-time discrete diffusion models.  
arXiv  
preprint arXiv:2211.16750  
, 2022.  
Suzgun, M., Scales, N., Sch  
¨  
arli, N., Gehrmann, S., Tay,  
Y., Chung, H. W., Chowdhery, A., Le, Q. V., Chi,  
E. H., Zhou, D., et al. Challenging big-bench tasks and  
whether chain-of-thought can solve them.  
arXiv preprint  
arXiv:2210.09261  
, 2022.  
12  
  
  
Large Language Diffusion Models  
Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi,  
A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P.,  
Bhosale, S., et al. Llama 2: Open foundation and ne-  
tuned chat models.  
arXiv preprint arXiv:2307.09288  
,  
2023.  
Uria, B., Murray, I., and Larochelle, H. A deep and tractable  
density estimator. In  
Proceedings of the 31th Interna-  
tional Conference on Machine Learning  
, 2014.  
Vaswani, A. Attention is all you need.  
arXiv preprint  
arXiv:1706.03762  
, 2017.  
Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J.,  
Chen, Z., Tang, J., Chen, X., Lin, Y., et al. A survey on  
large language model based autonomous agents.  
Frontiers  
of Computer Science  
, 18(6):186345, 2024a.  
Wang, X., Zheng, Z., Ye, F., Xue, D., Huang, S., and Gu, Q.  
Diffusion language models are versatile protein learners.  
arXiv preprint arXiv:2402.18567  
, 2024b.  
Wang, X., Zheng, Z., Ye, F., Xue, D., Huang, S., and Gu, Q.  
Dplm-2: A multimodal diffusion protein language model.  
arXiv preprint arXiv:2410.13782  
, 2024c.  
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi,  
E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting  
elicits reasoning in large language models.  
Advances in  
neural information processing systems  
, 35:24824Œ24837,  
2022.  
Wu, T., Fan, Z., Liu, X., Gong, Y., Shen, Y., Jiao, J., Zheng,  
H.-T., Li, J., Wei, Z., Guo, J., Duan, N., and Chen, W.  
Ar-diffusion: Auto-regressive diffusion model for text  
generation, 2023.  
Xu, C., Wang, X., Liao, Z., Li, Y., Hou, T., and Deng,  
Z. Show-o turbo: Towards accelerated unied multi-  
modal understanding and generation.  
arXiv preprint  
arXiv:2502.05415  
, 2025.  
Xue, K., Zhou, Y., Nie, S., Min, X., Zhang, X., Zhou, J.,  
and Li, C. Unifying bayesian ow networks and diff usion  
models through stochastic differential equations.  
arXiv  
preprint arXiv:2404.15766  
, 2024.  
Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B.,  
Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu,  
J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang,  
K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M.,  
Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Xia, T., Ren,  
X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y.,  
Cui, Z., Zhang, Z., and Qiu, Z. Qwen2.5 technical report.  
arXiv preprint arXiv:2412.15115  
, 2024.  
Ye, J., Zheng, Z., Bao, Y., Qian, L., and Gu, Q. Diffusion  
language models can perform many tasks with scaling and  
instruction-netuning.  
arXiv preprint arXiv:2308.12219  
,  
2023a.  
Ye, J., Zheng, Z., Bao, Y., Qian, L., and Wang, M. Dinoiser:  
Diffused conditional sequence learning by manipulating  
noises.  
arXiv preprint arXiv:2302.10025  
, 2023b.  
Ye, T., Xu, Z., Li, Y., and Allen-Zhu, Z. Physics of Lan-  
guage Models: Part 2.1, Grade-School Math and the Hid-  
den Reasoning Process.  
ArXiv e-prints  
, abs/2407.20311,  
July 2024. Full version available at  
http://arxiv.  
org/abs/2407.20311  
.  
Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi,  
Y. Hellaswag: Can a machine really nish your sentence?  
arXiv preprint arXiv:1905.07830  
, 2019.  
Zhang, B. and Sennrich, R. Root mean square layer nor-  
malization.  
Advances in Neural Information Processing  
Systems  
, 32, 2019.  
Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y.,  
Min, Y., Zhang, B., Zhang, J., Don g, Z., et al. A survey of  
large language models.  
arXiv preprint arXiv:2303.18223  
,  
2023.  
Zheng, K., Chen, Y., Mao, H., Liu, M.-Y., Zhu, J., and  
Zhang, Q. Masked diffusion models are secretly time-  
agnostic masked models and exploit inaccurate categor-  
ical sampling, 2024. URL  
https://arxiv.org/  
abs/2409.02908  
.  
Zheng, L., Yuan, J., Yu, L., and Kong, L. A reparameter-  
ized discrete diffusion model for text generation.  
ArXiv  
,  
abs/2302.05737, 2023.  
13  
  
  
Large Language Diffusion Models  
Algorithm 1  
Pre-training of LLaDA  
Require:  
mask predictor  
p  
  
, data distribution  
p  
data  
1:  
repeat  
2:  
x  
0  
˘  
p  
data  
,  
t  
˘  
U  
(0  
;  
1]  
# with a probability of 1%, the sequence length of  
x  
0  
follows U  
[1  
;  
4096]  
3:  
x  
t  
˘  
q  
t  
j  
0  
(  
x  
t  
j  
x  
0  
)  
#  
q  
t  
j  
0  
is dened in Eq. (  
7  
)  
4:  
Calculate  
L  
=  
  
1  
t  
  
L  
P  
L  
i  
=1  
1  
[  
x  
i  
t  
=  
M  
] log  
p  
  
(  
x  
i  
0  
j  
x  
t  
)  
#  
L  
is the sequence length of  
x  
0  
5:  
Calculate  
r  
  
L  
and run optimizer.  
6:  
until  
Converged  
7:  
Return  
p  
  
Algorithm 2  
Supervised Fine-Tuning of LLaDA  
Require:  
mask predictor  
p  
  
, pair data distribution  
p  
data  
1:  
repeat  
2:  
p  
0  
; r  
0  
˘  
p  
data  
,  
t  
˘  
U  
(0  
;  
1]  
# please refer to Appendix  
 B.1  
 for details on the SFT data processing.  
3:  
r  
t  
˘  
q  
t  
j  
0  
(  
r  
t  
j  
r  
0  
)  
#  
q  
t  
j  
0  
is dened in Eq. (  
7  
)  
4:  
Calculate  
L  
=  
  
1  
t  
  
L  
0  
P  
L  
0  
i  
=1  
1  
[  
r  
i  
t  
=  
M  
] log  
p  
  
(  
r  
i  
0  
j  
p  
0  
; r  
t  
)  
#  
L  
0  
is the sequence length of  
r  
0  
5:  
Calculate  
r  
  
L  
and run optimizer.  
6:  
until  
Converged  
7:  
Return  
p  
  
Algorithm 3  
Conditional Log-likelihood Evaluation of LLaDA  
Require:  
mask predictor  
p  
  
, prompt  
p  
0  
, response  
r  
0  
, the number of Monte Carlo estimations  
n  
mc  
1:  
log  
likelihood  
= 0  
2:  
for  
i  
   
1  
to  
n  
mc  
do  
3:  
l  
˘ f  
1  
;  
2  
; : : : ; L  
g  
#  
L  
is the sequence length of  
r  
0  
4:  
Obtain  
r  
l  
by uniformly sampling  
l  
tokens from  
r  
0  
without replacement for masking  
5:  
log  
likelihood  
=  
log  
likelihood  
+  
L  
l  
P  
L  
i  
=1  
1  
[  
r  
i  
l  
=  
M  
] log  
p  
  
(  
r  
i  
0  
j  
p  
0  
; r  
l  
)  
6:  
end for  
7:  
log  
likelihood  
=  
log  
likelihood  
=n  
mc  
8:  
Return  
log  
likelihood  
A. Formulation of Masked Diffusion Models  
A.1. Training  
Masked diffusion models (MDMs) (  
Austin et al.  
,  
 2021a  
;  
 Lou et al.  
,  
 2023  
;  
 Ou et al.  
,  
 2024  
) dene the model distribution  
p  
  
(  
x  
0  
)  
in a manner distinct from autoregressive models.  
These models introduce a forward process  
f  
x  
t  
g  
indexed by a time  
t  
2  
[0  
;  
1]  
. This process gradually and independently  
masks all tokens in the sequence  
x  
0  
. At time  
t  
= 0  
, the data point  
x  
0  
is fully observed with no masks, while for  
t  
2  
(0  
;  
1]  
,  
x  
t  
represents latent variables with varying mask ratios in expectation.  
Formally, the conditional distribution of  
x  
t  
given  
x  
0  
is dened by a fully factorized form:  
q  
t  
j  
0  
(  
x  
t  
j  
x  
0  
) =  
L  
Y  
i  
=1  
q  
t  
j  
0  
(  
x  
i  
t  
j  
x  
i  
0  
)  
;  
(7)  
where the conditional distribution for each token is given by:  
q  
t  
j  
0  
(  
x  
i  
t  
j  
x  
i  
0  
) =  
(  
1  
  
t; x  
i  
t  
=  
x  
i  
0  
;  
t; x  
i  
t  
=  
M  
:  
(8)  
Here,  
M  
denotes the mask token. Intuitively, each token either remains unchanged or is masked, with the probability of  
14  
  
  
Large Language Diffusion Models  
Algorithm 4  
Reverse Process of LLaDA  
Require:  
mask predictor  
p  
  
, prompt  
p  
0  
, answer length  
L  
, sampling steps  
N  
1:  
Set  
r  
1  
is a fully masked sequence of length  
L  
.  
2:  
for  
t  
   
1  
down to  
1  
N  
step  
1  
N  
do  
3:  
s  
=  
t  
  
1  
N  
4:  
r  
0  
= arg max  
r  
0  
p  
  
(  
r  
0  
j  
p  
0  
; r  
t  
)  
# we employ greedy sampling when predicting masked tokens  
5:  
for  
i  
   
1  
to  
L  
do  
6:  
if  
r  
t  
6  
=  
M  
then  
7:  
r  
i  
0  
=  
r  
i  
t  
8:  
else  
9:  
With probability  
s  
t  
,  
r  
i  
0  
is set to M  
10:  
end if  
11:  
end for  
12:  
r  
s  
=  
r  
0  
13:  
end for  
14:  
Return  
r  
0  
being masked increasing linearly as  
t  
progresses from  
0  
to  
1  
. At  
t  
= 1  
, all tokens are guaranteed to be masked, meaning that  
x  
1  
follows a Dirac distribution concentrated on a sequence of fully masked tokens. Notably, the linear masking probability  
is analogous to but distinct from, the noise schedule in continuous diffusion models (  
Sohl-Dickstein et al.  
,  
 2015  
;  
 Ho et al.  
,  
2020  
;  
 Song et al.  
,  
 2020  
). This linearity is motivated by the assumption that the information in the text is proportional to the  
number of tokens on average, making it reasonable to lose information linearly during the forward process.  
The forward process is not only reversible but also corresponds to a reverse process that is fully factorized across all  
tokens (  
Austin et al.  
,  
 2021a  
). The reverse process, from time  
t  
= 1  
to  
0  
, generates new data from sequences of fully masked  
tokens. The conditional distribution for the reverse process, for  
0  
  
s < t  
  
1  
, is factorized as:  
q  
s  
j  
t  
(  
x  
s  
j  
x  
t  
) =  
L  
Y  
i  
=1  
q  
s  
j  
t  
(  
x  
i  
s  
j  
x  
t  
)  
;  
(9)  
where the conditional distribution for each token is:  
q  
s  
j  
t  
(  
x  
i  
s  
j  
x  
t  
) =  
8  
>  
>  
>  
<  
>  
>  
>  
:  
1  
; x  
i  
t  
6  
=  
M  
; x  
i  
s  
=  
x  
i  
t  
;  
s  
t  
; x  
i  
t  
=  
M  
; x  
i  
s  
=  
M  
;  
t  
  
s  
t  
q  
0  
j  
t  
(  
x  
i  
s  
j  
x  
t  
)  
; x  
i  
t  
=  
M  
; x  
i  
s  
6  
=  
M  
;  
0  
;  
otherwise  
:  
(10)  
Thus, the key function to estimate is the conditional distribution  
q  
0  
j  
t  
(  
x  
i  
s  
j  
x  
t  
)  
, which predicts the original token if it is masked  
in the input  
x  
t  
. This is analogous to the  
data prediction  
form in continuous diffusion models.  
As proven in (  
Ou et al.  
,  
 2024  
), an equivalent yet  
time-free  
parameterization can be derived as:  
q  
0  
j  
t  
(  
x  
i  
s  
j  
x  
t  
) =  
p  
data  
(  
x  
i  
0  
j  
x  
UM  
t  
)  
;  
8  
i  
such that  
x  
i  
t  
=  
M  
;  
(11)  
where  
x  
UM  
t  
denotes the collection of unmasked tokens in  
x  
t  
, which is identical to the corresponding tokens in the original  
data  
x  
0  
since unmasked tokens are solely determined by  
x  
0  
and are independent of time  
t  
. Intuitively, this implies that  
estimating the data prediction function is equivalent to estimating the conditional distributions on clean data, which is  
time-invariant. Consequently, the time  
t  
need not be provided as input to the parametric model.  
Although the development of masked diffusion is nontrivial, the implementation is straightforward. We rst introduce the  
mask predictor  
, a parametric model  
p  
  
(  
j  
x  
t  
)  
(e.g., a Transformer without a causal mask), which takes  
x  
t  
for any  
t  
as input  
and predict all masked tokens simultaneously. Then, we dene the model distribution  
p  
  
(  
x  
0  
)  
as follows: starting with  
x  
1  
as  
a sequence of fully masked tokens, we simulate an approximate reverse process parameterized by  
p  
  
(  
j  
x  
t  
)  
from  
t  
= 1  
to  
0  
.  
The marginal distribution induced at  
t  
= 0  
then represents the model distribution  
p  
  
(  
x  
0  
)  
.  
15  
  
  
Large Language Diffusion Models  
Algorithm 5  
Low-condence Remasking Strategy of LLaDA  
Require:  
mask predictor  
p  
  
, prompt  
p  
0  
, answer length  
L  
, sampling steps  
N  
1:  
Set  
r  
1  
is a fully masked sequence of length  
L  
.  
2:  
for  
t  
   
1  
down to  
1  
N  
step  
1  
N  
do  
3:  
s  
=  
t  
  
1  
N  
4:  
for  
i  
   
1  
to  
L  
do  
5:  
if  
r  
i  
t  
6  
=  
M  
then  
6:  
r  
i  
0  
=  
r  
i  
t  
,  
c  
i  
= 1  
7:  
else  
8:  
r  
i  
0  
= arg max  
r  
i  
0  
p  
  
(  
r  
i  
0  
j  
p  
0  
; r  
t  
)  
9:  
c  
i  
=  
p  
  
(  
r  
i  
0  
j  
p  
0  
; r  
t  
)  
r  
i  
0  
10:  
end if  
11:  
end for  
12:  
n  
un  
=  
b  
L  
(1  
  
s  
)  
c  
# the number of unmasked tokens is  
n  
un  
in timestep  
s  
13:  
for  
i  
   
1  
to  
L  
do  
14:  
if  
c  
i  
2  
Lowest  
  
n  
un  
  
f  
c  
i  
g  
L  
1  
  
then  
15:  
r  
i  
0  
=  
M # the  
n  
un  
positions with the least condence are selected for remasking.  
16:  
end if  
17:  
end for  
18:  
r  
s  
=  
r  
0  
19:  
end for  
20:  
Return  
r  
0  
Formally, the mask predictor is trained using a cross-entropy loss with masking:  
L  
(  
  
)  
,  
  
E  
t;x  
0  
;x  
t  
"  
1  
t  
L  
X  
i  
=1  
1  
[  
x  
i  
t  
=  
M  
] log  
p  
  
(  
x  
i  
0  
j  
x  
t  
)  
#  
;  
(12)  
where  
x  
0  
is sampled from the training data,  
t  
is sampled uniformly from  
[0  
;  
1]  
, and  
x  
t  
is sampled from  
q  
t  
j  
0  
(  
x  
t  
j  
x  
0  
)  
. The  
indicator function  
1  
[  
  
]  
ensures that the cross-entropy loss is computed only for masked tokens. In  
 Ou et al.  
 (  
2024  
), it has  
been proven that the loss function  
L  
(  
  
)  
is an upper bound on the negative log-likelihood of the model distribution:  
  
E  
x  
0  
˘  
p  
data  
(  
x  
0  
)  
[log  
p  
  
(  
x  
0  
)]  
 L  
(  
  
)  
:  
(13)  
In summary, this principled approach trains a generative model by progressively masking tokens during a forward process  
and learning to recover the data distribution during a reverse process, all under the (approximate) maximum likelihood  
estimation framework.  
A.2. Inference  
The cross-entropy loss in Eq. (  
12  
) has several equivalent forms (  
Ou et al.  
,  
 2024  
). The rst one is given by  
  
E  
l  
˘f  
1  
;  
2  
;:::;L  
g  
;x  
0  
;x  
l  
"  
L  
l  
L  
X  
i  
=1  
1  
[  
x  
i  
l  
=  
M  
] log  
p  
  
(  
x  
i  
0  
j  
x  
l  
)  
#  
;  
(14)  
where  
l  
is uniformly sampled from  
f  
1  
;  
2  
; : : : ; L  
g  
, and  
x  
l  
is obtained by uniformly sampling  
l  
tokens from  
x  
0  
without  
replacement for masking. Despite masking exactly  
l  
tokens is different from masking each token independently with  
probability  
t  
, these two masking methods lead to equivalent results in expectation (  
Ou et al.  
,  
 2024  
).  
While Eq. (  
12  
) and Eq. (  
14  
) share the same expectation, their variances differ. Intuitively, in Eq. (  
12  
), we expect  
x  
t  
to  
have a fraction of  
t  
tokens masked. However, the randomness of the forward process (i.e., Eq. (  
7  
)) often causes deviations,  
especially when  
x  
t  
contains few tokens. In contrast, in Eq. (  
14  
), the fraction of masked tokens in  
x  
l  
is deterministically  
l  
L  
.  
While a theoretical analysis depends on the data distribution, empirical results show that Eq. (  
12  
) requires over 1000 Monte  
Carlo estimates for stable results, whereas Eq. (  
14  
) achieves stability with only 128 estimates. In addition, we can simply  
modify Eq. (  
14  
) to its conditional version (i.e., Eq. (  
6  
)) based on Eq. (  
5  
).  
16  
  
  
Large Language Diffusion Models  
Figure 4.  
A Conceptual Overview of the Semi-autoregressive Sampling.  
Any-order autoregressive models (AO-ARM) (  
Uria et al.  
,  
 2014  
;  
 Hoogeboom et al.  
,  
 2021a  
;  
 Shih et al.  
,  
 2022  
) characterize the  
joint distribution autoregressively for all possible orders  
ˇ  
of the  
L  
variables. To learn such a distribution, an AO-ARM  
utilizes a weight-sharing neural network to model all univariate conditionals and employs mask tokens to represent absent  
variables. During training, the expected negative log-likelihood over the uniform distribution of all orders  
U  
ˇ  
is minimized:  
  
E  
x  
0  
;ˇ  
˘  
U  
ˇ  
"  
L  
X  
i  
=1  
log  
p  
  
(  
x  
ˇ  
(  
i  
)  
0  
j  
x  
ˇ  
(  
<i  
)  
0  
;  
ˇ  
)  
#  
:  
(15)  
Intuitively,  
x  
ˇ  
(  
<i  
)  
0  
can be understood as a masked token  
x  
t  
with index in  
ˇ  
(  
  
i  
)  
being masked. It can be further proved that  
Eq. (  
15  
) is equivalent to Eq. (  
12  
). This connection explains the bidirectional reasoning capabilities of LLaDA, even though  
it was never used explicitly in the inference procedure.  
Nie et al.  
 (  
2024  
) introduce unsupervised classier-free guidance, a plug-and-play technique that balances alignment with  
prompts and text diversity. Specically, unsupervised classier-free guidance employs the following modied mask predictor  
for inference:  
~  
p  
  
(  
r  
0  
j  
p  
0  
; r  
t  
)  
/  
p  
  
(  
r  
0  
j  
p  
0  
; r  
t  
)  
1+  
w  
p  
  
(  
r  
0  
j  
m; r  
t  
)  
w  
;  
(16)  
where  
m  
is a mask sequence of the same length as  
p  
0  
and  
w  
is a hyperparameter that controls the strength of  
p  
0  
. We employ  
unsupervised classier-free guidance in the downstream tasks, as detailed in Appendix  
 B.5  
.  
A.3. Algorithms  
In this section, we present the training and inference algorithms. Specically, we introduce the pre-training and supervised  
ne-tuning algorithms in Algorithm  
 1  
 and Algorithm  
 2  
, respectively. In addition, the likelihood evaluation algorithm is  
provided in Algorithm  
 3  
. Finally, we present the sampling algorithms in Algorithm  
 4  
 and Algorithm  
 5  
, which correspond to  
the original reverse process (i.e., random remasking) and the low-condence remasking strategy (  
Chang et al.  
,  
 2022  
) of  
LLaDA, respectively.  
Furthermore, we apply a semi-autoregressive remasking strategy for the instruct model. Specically, the sequence is  
divided into several blocks and generated from left to right. Within each block, we apply the random remasking or the  
low-condence remasking strategy for sampling. For clarity and better visualization, we illustrate the semi-autoregressive  
sampling process in Fig.  
 4  
.  
Our semi-autoregressive remasking strategy is inspired by the prior work of  
 Anonymous  
 (  
2025  
). However, it is important to  
note that  
 Anonymous  
 (  
2025  
) and LLaDA have entirely different training objectives and network architectures. Specically,  
the pre-training objective of  
 Anonymous  
 (  
2025  
) differs from Eq. (  
3  
). Moreover, their approach incorporates a causal mask  
in the Transformer attention block, whereas LLaDA employs a bidirectional Transformer.  
17  
  
  
Large Language Diffusion Models  
Table 5.  
Model Architecture.  
We provide the congurations of our 1B and 7B ARM baselines, as well as the 1B and 8B LLaDA and the  
8B LLaMA3 models.  
Our ARM Baseline 1B LLaDA 1B Our ARM Baseline 7B LLaDA 8B LLaMA3 8B  
Layers 22 22 28 32 32  
Model dimension 2048 2048 4096 4096 4096  
Attention heads 32 32 32 32 32  
Vocabulary size 126,464 126,464 126,464 126,464 128,000  
FFN dimension 5634 5634 13,440 12,288 14,336  
Key/Value heads 4 4 8 32 8  
Total parameters 1.49 B 1.49 B 6.83 B 8.02 B 8.03 B  
Non-embedding parameters 0.97 B 0.97 B 5.80 B 6.98 B 6.98 B  
B. Experiments  
B.1. Details of SFT Data  
A dynamic sequence length is employed during supervised SFT, where  
j  
EOS  
j  
tokens are appended to the end of short pairs  
to ensure equal sequence lengths across all data in each mini-batch. Notably, the padding  
j  
EOS  
j  
tokens are treated as part of  
the response, i.e., masked and included in the training objective. The  
j  
EOS  
j  
tokens are removed from the generated outputs  
during sampling. This strategy is crucial for LLaDA and ensures that the model learns to control the length of its responses  
by generating  
j  
EOS  
j  
, enabling the response length to align effectively with the given prompt.  
In addition, for  
n  
-turn dialogues  
(  
p  
0  
0  
; r  
0  
0  
; p  
1  
0  
; r  
1  
0  
; : : : ; p  
n  
  
1  
0  
; r  
n  
  
1  
0  
)  
, we treat it as  
n  
single-turn dialogue pairs, i.e.,  
(  
p  
0  
0  
; r  
0  
0  
)  
;  
(  
p  
0  
0  
r  
0  
0  
p  
1  
0  
; r  
1  
0  
)  
; : : : ;  
(  
p  
0  
0  
r  
0  
0  
p  
1  
0  
r  
1  
0  
: : : p  
n  
  
1  
0  
; r  
n  
  
1  
0  
)  
and randomly sample one. This data partitioning strategy not only  
equips LLaDA with multi-turn dialogue capabilities but also aligns with the padding strategy.  
B.2. Architectures and Scaling Experiments  
This section provides detailed information about the models we trained and the scalability experiments described in Sec.  
 3.1  
.  
Firstly, for efciency, we trained an ARM and a MDM, both with 1.5B parameters and identical architectures. Additionally,  
we scaled the MDM to 8B parameters. Due to computational resource constraints, we did not train an 8B autoregressive  
model with the same architecture. Instead, we utilized our previously trained 7B autoregressive model for comparison.  
These four models are utilized in the scalability analysis in Sec.  
 3.1  
.  
We adopted a Transformer architecture similar to LLaMA (  
Touvron et al.  
,  
 2023  
;  
 Dubey et al.  
,  
 2024  
) for the ARMs and MDMs  
we trained. Specically, we employ RMSNorm (  
Zhang & Sennrich  
,  
 2019  
) to stabilize training, used SwiGLU (  
Shazeer  
,  
2020  
) as the activation function to enhance non-linearity, and integrate RoPE (  
Su et al.  
,  
 2024  
) for more expressive positional  
encoding. Tab.  
 5  
 provides an overview of the model architectures.  
Additionally, we employ the widely used  
6  
N D  
formulation (  
Kaplan et al.  
,  
 2020  
;  
 Hoffmann et al.  
,  
 2022  
) to calculate the  
training FLOPs in Fig.  
 3  
, where  
N  
represents the number of non-embedding parameters, and  
D  
denotes the total number of  
training tokens.  
B.3. Details and Ablation on Remasking  
This section rst explains the inference methods in detail, followed by ablation studies on various remasking strategies.  
For instruct model, since our training includes a large number of  
j  
EOS  
j  
tokens, directly applying the lowest condence  
remasking strategy would result in an excessively high and unnatural occurrence of  
j  
EOS  
j  
tokens in the generated sentences.  
Therefore, we divide the entire generation length into several blocks. Each block is generated from left to right, and within  
each block, we apply the reverse process to perform sampling. However, for the base model, since  
j  
EOS  
j  
tokens are not  
padded during training, using the semi-autoregressive remasking strategy makes little difference in the results.  
18  
  
  
Large Language Diffusion Models  
Table 6.  
Ablation on Remasking.  
We use GSM8K accuracy as the evaluation metric. The lowest-condence & semi-autoregressive  
remasking strategy refers to generating each block from left to right, while applying the lowest-condence remasking strategy within each  
block.  
LLaDA 8B Base LLaDA 8B Instruct  
Randomly remasking 52.3 72.0  
Lowest condence remasking  
64.7  
12.9  
Lowest condence & semi-autoregressive remasking 64.4  
73.8  
Table 7.  
Ablation on Generated Length.  
We use the GSM8K accuracy as the evaluation metric and set the number of sampling steps to  
half of the generated length.  
Length LLaDA 8B Base LLaDA 8B Instruct  
256 62.5 75.3  
512 64.7 73.8  
1024 65.9 75.3  
For efciency, we set the answer length to 512 and the number of sampling steps to 256 in our ablation studies. For the  
semi-autoregressive remasking strategy, we set the block length to 32. To simplify the evaluation, we use GSM8K accuracy  
as the performance metric. As shown in Table  
 6  
, for the base model, applying the lowest condence remasking strategy  
signicantly improves performance, whereas the semi-autoregressive remasking strategy does not yield any noticeable  
benets. In contrast, for the instruct model, directly applying the lowest condence remasking strategy leads to a substantial  
performance drop. However, combining the lowest condence remasking strategy with the semi-autoregressive remasking  
strategy results in the best performance for the instruct model.  
We provide the specic remasking strategies and hyperparameters used for each benchmark in Appendix  
 B.5  
.  
B.4. Ablation on Generated Length  
In this section, we conduct ablation studies on the generated length. For simplicity, we use GSM8K as the evaluation metric.  
To ensure efciency and fairness, for each setting, we set the number of sampling steps to half of the generated length,  
ensuring that in each sampling step, two tokens are transferred from the mask to the text.  
For the LLaDA 8B Base model, we employ the lowest-condence remasking strategy (i.e., Algorithm  
 5  
). For the LLaDA  
8B Instruct model, we adopt semi-autoregressive sampling with a block length of 32.  
As reported in Tab.  
 7  
, both the Base model and the Instruct model exhibit minimal sensitivity to the length hyperparameter.  
B.5. Standard Benchmarks and Evaluation  
In this section, we briey introduce the benchmarks used in our study and describe the details of our evaluation process.  
Following standard LLM (  
Chu et al.  
,  
 2024  
;  
 Yang et al.  
,  
 2024  
) evaluation practices, we assess LLaDA across four dimensions:  
General ability:  
MMLU (  
Hendrycks et al.  
,  
 2020  
), BBH (  
Suzgun et al.  
,  
 2022  
), ARC-C (  
Clark et al.  
,  
 2018  
), Hel-  
laswag (  
Zellers et al.  
,  
 2019  
), TruthfulQA (  
Lin et al.  
,  
 2021  
), WinoGrande (  
Sakaguchi et al.  
,  
 2021  
) and PIQA (  
Bisk  
et al.  
,  
 2020  
).  
Math and science ability:  
GSM8K (  
Cobbe et al.  
,  
 2021  
), Math (  
Hendrycks et al.  
,  
 2021  
) and GPQA (  
Rein et al.  
,  
 2023  
).  
Code generation:  
HumanEval (  
Chen et al.  
,  
 2021  
), HumanEval-FIM (  
Bavarian et al.  
,  
 2022  
) and MBPP (  
Austin et al.  
,  
2021b  
).  
Chinese understanding:  
CMMLU (  
Li et al.  
,  
 2023  
) and C-Eval (  
Huang et al.  
,  
 2024a  
).  
For all the aforementioned benchmarks, we follow the widely adopted evaluation process (  
Gao et al.  
,  
 2024  
) used in LLM  
assessments, primarily employing conditional likelihood estimation and conditional generation. Specically, for certain  
benchmarks, a prompt and multiple candidate answers are provided, and the model is required to compute each candidate's  
19  
  
  
Large Language Diffusion Models  
Table 8.  
Inference Conguration for LLaDA 8B Instruct.  
We apply the low-condence remasking strategy across all datasets. A block  
length smaller than the answer length indicates the use of the semi-autoregressive remasking strategy; otherwise, it is not applied. The  
shorter answer lengths for MMLU and HellaSwag result from their requirement to generate only a single token (i.e., A, B, C, or D) and  
two special end tokens (i.e., end of the model response and end of the sentence).  
Answer length Block length  
MMLU 3 3  
Hellaswag 3 3  
GPQA 128 64  
GSM8K 256 8  
HumanEval 512 32  
MBPP 512 32  
Math 256 256  
MMLU-pro 256 256  
ARC-C 512 512  
Table 9.  
Comparison on iGSM Dataset.  
4 steps 5 steps 6 steps  
LLaMA3 8B Base 38.0 35.0 34.0  
LLaDA 8B Base  
64.0 41.0 44.0  
conditional likelihood. The candidate with the highest likelihood is then selected as the model's nal answer, and accuracy  
is used as the evaluation metric. For the remaining benchmarks, the model generates responses based on the given prompt,  
and performance is evaluated using metrics such as exact match and other relevant criteria.  
For the base model, we use conditional likelihood estimation for MMLU, CMMLU, C-Eval, ARC-C, Hellaswag, TruthfulQA,  
WinoGrande, PIQA, and GPQA, while the remaining benchmarks are evaluated using conditional generation. For the  
instruct model, we evaluate all benchmarks using conditional generation. For datasets evaluated with conditional likelihood  
estimation, we adopt the widely used lm-evaluation-harness (  
Gao et al.  
,  
 2024  
) framework. However, for datasets evaluated  
with conditional generation, we rely o n an internal library for evaluation, as lm-evaluation-harness lacks implementations  
for certain key metrics (e.g., HumanEval-FIM).  
For benchmarks evaluated using conditional likelihood estimation, we use Monte Carlo estimation to approximate Eq. (  
6  
)  
for conditional likelihood estimation. Since MMLU, CMMLU, and C-EVAL only require the likelihood of a single token, a  
single Monte Carlo estimate is sufcient for these benchmarks. For all other benchmarks, we nd that 128 Monte Carlo  
samples are adequate to produce stable results. For ARC-C, Hellaswag, TruthfulQA, WinoGrande, PIQA, and GPQA, we  
report the results with the optimal unsupervised classier-free guidance scale searched in  
f  
0  
;  
0  
:  
5  
;  
1  
;  
1  
:  
5  
;  
2  
g  
.  
For benchmarks evaluated using conditional generation, we apply the lowest condence remasking strategy to the base model,  
setting both the response length and the number of sampling steps to 1024. For the instruct model, we set the sampling steps  
equal to the answer length and provide the inference hyperparameters in Table  
 8  
. Due to computational resource constraints,  
we do not use the unsupervised classier-free guidance for these benchmarks evaluated using conditional generation.  
B.6. Analysis of Sampling Steps  
In this section, we analyze the impact of sampling steps. For simplicity, we use the LLaDA 8B Base model and evaluate it  
on the GSM8K and HumanEval datasets. Additionally, we x the answer length at 1024. As shown in Fig.  
 5  
, as the number  
of sampling steps increases to match the answer length, the model's accuracy improves accordingly.  
B.7. Evaluation on iGSM Dataset  
To further assess the mathematical capabilities of LLaDA, we test its performance on iGSM (  
Ye et al.  
,  
 2024  
), an innite,  
synthetic GSM8K-like dataset. iGSM is generated via specic rules, with parameters that control the dif culty of problems  
(i.e., the number of solution steps). For evaluation consistency, we append ﬂ####  
$  
answerﬂ to the nal solution, adhering to  
20  
  
  
Large Language Diffusion Models  
Figure 5.  
Analysis of Sampling Steps.  
the GSM8K format. Below is an example with solution steps set to 4:  
(  
Question  
) The number of each North Star Elementary's Cultural Studies Classroom equals 1. The number of each  
Westridge Elementary's Dance Studio equals 3 times as much as the sum of each North Star Elementary's Classroom  
and each North Star Elementary's Cultural Studies Classroom. How many Dance Studio does Westridge Elementary  
have?  
(  
Solution  
) Dene North Star Elementary's Cultural Studies Classroom as x; so x = 1.  
Dene North Star Elementary's Classroom as m; so m = x = 1.  
Dene Westridge Elementary's Dance Studio as n; w = m + x = 1 + 1 = 2;  
so n = 3 * w = 3 * 2 = 1 #### 1  
Since there are slight differences between GSM8K and iGSM (e.g., the use of a mod 5 algorithmic system), we follow (  
Ye  
et al.  
,  
 2024  
) and provide a system prompt along with four-shot question-answer pairs for each problem.  
(  
Prompt  
) You're an expert at solving elementary math problems involving addition, subtraction, and multiplication.  
You solve all the problems in a uniform format. All calculations are done modulo 5. For example, 4 + 4 equals 3, 2 + 4  
equals 1, 3 + 3 + 3 equals 4, 3 * 3 equals 4, and 2 * 2 equals 4. When providing your solution, please end with '#### x.'  
where x is your nal answer, an integer between 0 and 4. You must solve all the problems using the same solution  
format. Our scenarios involve up to four categories of objects: schools, classrooms, backpacks and stationeries. Each  
school may contain classrooms, each classroom may contain backpacks, and each backpack may contain stationeries.  
We can specify quantities, such as  
¨  
the number of dance studios at each Lakeshore High.  
¨  
Assume that every entity  
with the same name has an identical conguration; for example, each Lakeshore High contains the same number of  
dance studios. Another guiding principle is that what is not mentioned does not exist: when we refer to classrooms at  
Lakeshore High, we are only discussing the classrooms explicitly mentioned in our scenario. Furthermore, if Lakeshore  
High is not even mentioned, any classroom within it is automatically considered to be non-existent (i.e. 0).  
For solution steps ranging from 4 to 6, we generate 100 questions for each case and report the corresponding accuracy in  
Table  
 9  
. As shown in the table, LLaDA Base demonstrates signicant and consistent advantages over LLaMA3 on unseen  
mathematical problems, aligning with the results in Table  
 1  
.  
B.8. Poem Completion Tasks  
In this section, we present examples from our poem completion dataset as follows.  
Example 1:  
Prompt:  
‹Ł  
Ñ  
s  
—  
  
å  
/  
ÀH  
˜  
ô  
¥  
ﬁ  
ú  
å  
P  
sï  
  
Answer:  
˙  
P  
}  
  
  
21  
  
  
Large Language Diffusion Models  
Table 10.  
Visualization of the Semi-Autoregressive Sampling Process.  
In the response of LLaDA, darker colors indicate tokens  
predicted in the later stages of sampling, while lighter colors correspond to earlier predictions.  
User What are the benets of regular exercise for physical and mental health?  
LLaDA  
Table 11.  
Visualization of the Multi-turn Dialogue.  
We employ random remasking strategy. In the response of LLaDA, darker colors  
indicate tokens predicted in the later stages of sampling, while lighter colors correspond to earlier predictions.  
User  
I want to have a pet, but I don't know whether to get a cat or a dog. Do you have any  
suggestions?  
LLaDA  
User  
I'm very busy at work, so it seems like a cat would be a more suitable o ption. Do you have  
any suggestions for getting a cat?  
LLaDA  
Example 2:  
Prompt:  
  
Ø  
  
<  
M  
º  
M  
—  
  
å  
/  
ÀH  
˜  
ô  
¥  
ﬁ  
ú  
å  
P  
sï  
  
Answer:  
  
š  
)  
l  
Í  
Œ  
Þ  
  
B.9. More Case Studies  
In this section, we present additional case studies of LLaDA 8B Instruct. First, Tab.  
 10  
 illustrates the sampling process of  
semi-autoregressive remasking, while Tab.  
 11  
 depicts the sampling process for multi-turn dialogues with random remasking.  
Additionally, Tab.  
 12  
 and Tab.  
 13  
 provide further examples of single-turn and multi-turn dialogues. Finally, Tab.  
 14  
 presents  
examples of poem reversal completions where the LLaDA 8B Instruct model succeeds, in contrast to the failure of GPT-4o.  
22  
  
  
Large Language Diffusion Models  
Table 12.  
Single-turn Dialogue Cases of LLaDA 8B Instruct.  
Daily life  
User What's a piece of advice you've received that you've found helpful?  
LLaDA  
One helpful piece of advice I've received is to always be open to learning and growth.  
I've found that being willing to explore new ideas and experiences has helped me become  
more resilient and adaptable. By embracing a growth mindset, I've been able to identify  
opportunities for improvement and development, both personally and professionally.  
Daily life  
User Please recommend me three famous movies  
LLaDA  
Certainly! Here are three famous movies that have achieved widespread popularity and  
critical acclaim:  
1. **The Shawshank Redemption** (1994) - Inspired by a novella and directed by Frank  
Darabont, this lm tells the story of Andy Dufresne, a man convicted of murdering his  
wife, dealing with his harsh life in Shawshank Prison and forming an unlikely friendship  
with a fellow inmate.  
2. **The Godfather** (1972) - Directed by Francis Ford Coppola, this lm is based  
on Mario Puzo's novel of the same name. It follows the story of the Corleone family,  
particularly Michael Corleone, who rises to power and takes over the family's criminal  
empire.  
3. **The Empire Strikes Back** (1980) - Directed by George Lucas, this lm continues  
the story of a young Skywalker in the Star Wars saga. It follows Luke Skywalker as  
he helps the Rebel Alliance ght against the Galactic Empire and destroy the powerful  
weapon, the Death Star.  
All of these lms are celebrated for their cinematic achievements, powerful storytelling,  
and memorable characters.  
Translation  
User Please help me translate into Chinese: `What is now proved was once only imagined'  
LLaDA  
°  
(  
«  
Á  
  
—  
þ  
Ï  
ê  
/  
ó  
a  
Translation  
User Please help me translate into German: `What is now proved was once only imagined'  
LLaDA Wer heute best  
¨  
atigt ist, war einst nur immaginiert.  
Math  
User  
Emma decides to run 4 sprints 5 times a week. She runs 75 meters each sprint. How many  
total meters does she run a week?  
LLaDA  
Emma runs 4 sprints each time, so she runs a total of 4 x 5 = 20 sprints a week. Each sprint  
is 75 meters long, so Emma runs a total of 20 x 75 = 1500 meters a week. The nal result  
is 1500  
Code  
User I am writing python code, I have an array, please help me sort it from largest to smallest  
LLaDA  
Certainly! You can sort an array from the largest to smallest in Python using the `sort()'  
method or the `sorted()` function. Here's an example:  
### Using `sort()' method:  
my  
array = [3, 1, 4, 1, 5, 2]  
my  
array.sort(reverse=True)  
print(my  
array)  
23  
  
  
Large Language Diffusion Models  
Table 13.  
Multi-turn Dialogue Cases of LLaDA 8B Instruct.  
Daily life  
User How do I prepare for a job interview?  
LLaDA  
To prepare for a job interview, you should research the company and the job, practice  
common interview questions, and dress appropriately.  
User What are some common interview questions?  
LLaDA  
Some common interview questions include ﬁTell me about yourself ﬂ and ﬁWhy do you  
want to work for usﬂ?  
Daily life  
User How do I start learning programming?  
LLaDA  
To learn programming, you can start by choosing a programming language that you are  
interested in. Some popular languages include Python, Java, and C++.  
User Which programming language should I learn rst?  
LLaDA  
It depends on your interests and goals. Python is a great language for beginners for its  
simplicity and versatility.  
User How long does it take to learn Python?  
LLaDA It can take anywhere from a few weeks to several months to learn Python.  
User What are some good Python projects for beginners?  
LLaDA  
Some good Python projects for beginners include building a simple calculator, a game, and  
a web application.  
Math  
User  
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.  
How many clips did Natalia sell altogether in April and May?  
LLaDA  
Natalia sold clips to 48 friends in April. She sold half as many clips in May, so she sold  
48/2 = 24 clips in May. Altogether, Natalia sold 48 + 24 = 72 clips in April and May. The  
nal result is 72.  
User  
Natalia sold three times as many items in June as she did in May. How many clips did  
Natalia sell altogether in these three months?  
LLaDA  
Natalia sold clips to 48 friends in April. She sold half as many clips in May, so she sold  
48/2 = 24 clips in May. Natalia sold three times as many items in June as she did in May,  
so she sold 24 * 3 = 72 clips in June. Altogether, Natalia sold 48 + 24 + 72 = 144 clips in  
these three months. The nal result is 144.  
Table 14.  
Poem Reversal Completion Cases where LLaDA 8B Instruct Succeeds but GPT-4o Fails.  
Question1  
User `  
F  
û  
º  
í  
Í  
'  
—  
  
å  
/  
ÀH  
?  
GPT-4o  
ﬁ  
F  
û  
º  
í  
Í  
ﬂ  
—  
  
   
å  
/  
ﬁ  
à  
ï  
H  
U  
±  
=  
»  
ﬂ,  
Ù  
$  
å  
ú  
ê  
  
ã  
Í  
º  
N  
  
g  
—  
  
f  
u  
%  
  
% Z  
:  
à  
ï  
H  
U  
±  
=  
»  
  
F  
û  
º  
í  
Í  
  
Ù  
Œ  
Í  
h  
¾  
ƒ  
\  
  
(  
%  
)  
ö  
‡  
—  
d  
ì  
  
˜  
$  
  
˜  
±  
=  
„  
º  
í  
ð  
—  
Ï  
Ž  
  
Í  
   
ú  
Í  
º  
–  
Ã  
—  
à  
²  
1  
=  
  
LLaDA ﬁ  
F  
û  
º  
í  
Í  
ﬂ  
—  
  
å  
/  
ﬁ  
z  
q  
  
Á  
º  
ﬂ  
Question2  
User `  
˝  
Y  
%  
q  
z  
'  
—  
  
å  
/  
ÀH  
?  
GPT-4o  
ﬁ  
˝  
Y  
%  
q  
z  
ﬂ  
—  
  
   
å  
/  
ﬁ  
z  
q  
°  
è  
  
ﬂ.  
Ù  
$  
å  
ú  
ê  
  
ã  
×  
º  
‰  
ô  
—  
  
q E  
Ë  
œ  
	  
:  
z  
q  
°  
è  
  
  
˝  
Y  
%  
q  
z  
  
Ù  
Œ  
×  
Ï  
Ø  
ƒ  
%  
)  
q  
-  
è  
  
—  
Â  
Y  
  
z  
u  
  
h  
°  
ú  
ê  
6  
—  
†  
Y  
  
„  
  
”  
  
LLaDA ﬁ  
˝  
Y  
%  
q  
z  
ﬂ  
—  
  
å  
/  
ﬁ  
º  
ò  
B  
±  
=  
ﬂ  
24  
