# CUDARule Scoring & Optimization
What started as a hobby project in Q4 of 2024 has grown into a full fledged project. With a lot of support from the community I've built a tool that can score rules, counting the potential founds between a wordlist+rules and a target wordlist.

This tool presents a comprehensive package to score rules and optimize rulefiles, finding noise in the madness of rules. Prior to this rule it was not possible to easily determine if a set of rules was effective, but using the score functionality you can accurate determine how many founds a specific rule with have using a chosen wordlist. To this end I recommend using HashMob wordlists as basis.

HashMob is a password recovery community that launched back in 2021 and has since gained a large number of followers and active contributors. A weekly wordlist compromised of over 10,000 data breaches helps provide high quality plaintexts in near- realtime and Lethologica, the wallet recovery company funding this project has helped to grow the community into what it is today. Because of the large amount of statistically aggregated data (following the law of large numbers) we highly recommend using the HashMob huge or HashMob Combined Full as your 'target' wordlist when using this tool.
This will not only ensure that your ruleset is optimized to reflect real-world data, but also that it is statistically relevant. HashMob huge for example only takes the plaintexts that occur in >2 wordlists from over 10,000 data breaches -> A statistically significant dataset. 

As base wordlist for the research published during PasswordsCon 2025 I used the HashMob Large wordlist which bases its contained words on plaintexts that occur in >5 hashlists.

The program is based on CUDA SIMD programming and requires the use of NVCC to compile the kernel. Primary development happened using Cakes' hashcat docker container developed for Team HashMob with the use of Vast.AI and users utilizing windows might fight an increased difficulty in getting the program functional, I appologize for this in advance and welcome any PR with instructions to get Windows operational.


## Linux (Ubuntu) Quick-Start
To get started in linux Golang must be installed since the program is built in the language Go, with CGO as bridge to the CUDA kernel. Additionally, the CUDA toolkit must be installed to compile the kernel.
```bash
# Ensure your system is up to date.
apt update -y && apt dist-upgrade -y
# Install GoLang. The version below shows the latest version
wget https://go.dev/dl/go1.25.1.linux-amd64.tar.gz
rm -rf /usr/local/go && tar -C /usr/local -xzf go1.25.1.linux-amd64.tar.gz
# Install the NVIDIA Toolkit
sudo apt install -y nvidia-cuda-toolkit

# Add the following variables to ~/.profile to ensure the required Go & NVIDIA libraries can be found
export PATH=/usr/local/nvidia/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/go/bin:$PATH
export CPATH=/usr/local/cuda/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Then run the following to load the environment variables from ~/.profile:
source ~/.profile

# Compile the project using the build.sh or individual commands.
./build.sh
# or
nvcc --shared -o librules.so rules.cu -Xcompiler "-fPIC" --cudart static -arch=sm_80
go build -ldflags="-r . -s -w"
```

# Using the program
The program consists of two main phases, a scoring, and an optimizing phase. Finally, the optimized scores must be stripped of the scores in order for it to become a final usable hashcat rule-file.

These phases are connected and linked to each-other and any change in the input or target wordlists can result in unreliable results or unexpected behavior. 
### Requirements
Using the program requires a machine with the CUDA development kit installed and a compatible (NVIDIA) GPU, a set of words you want to apply rules to, the rules you wish to apply and optimize, and a dataset you want to compare against.
Choosing your wordlist and target dataset is vital to the quality of the results. Choosing a dataset that is too small or too different will not enable the rules to work their magic.

Therefore, it is recommended that you take your favorite wordlist and use either HashMob Huge or HashMob Combined full as target. With 13,000+ databreaches and billions of passwords you will get high quality results. From this 'base' scenario you can further explore alternative commands using your own datasets.

#### Choosing a dataset
When choosing your dataset there are a few considerations to make.
1. Size matters! Not only for the input wordlist you'll be using in attacks, but also the target wordlist. 
   1. The more words in the input wordlist, the larger the keyspace and the longer each individual rule will take to compute. At the same time it also means that the smaller wordlist you get the more quality of the wordlist has an impact.
   2. The more words in the target wordlist, the larger the memory and storage requirements. You cannot optimize against a target set that doesn't fit comfortably in your VRAM. As long as the dataset fits in your memory for optimize it is recommended you take the largest possible solution.

#### Memory usage
When talking about memory usage we try to limit the amount used as much as possible, but there are no way around some requirements. This means that fitting your dataset might require a certain amount of GPU VRAM. The easiest way to calculate a close estimate is to use the following formula:
```total memory usage = wordlistCount*5 + targetCount*5```
This will not be perfect as rules will have to be loaded in as well but becomes a close estimate.

### Phase 1 — Scoring
The first phase has the goal of scoring the performance of rules as if it was the first rule ran. This helps prioritize rules over others and enables larger optimizations later. This phase can be ran on as many rules as you wish and you can cut-off or remove low-scoring entries as you desire as long as the input and target wordlist remain unadjusted.
### Phase 2 — Optimizing
This phase will follow the scoring phase once you've scored all rules you want to evaluate. It is recommended to create a single file containing all scores and sorting them with the following command:
```
cat *.score | sort -rn > all_scores.dat
```
Then run the optimize command to start optimizing. It will work through several phases before starting the optimize process, and there are a few points to be mindful of.
1. A pause and resume function is available, allowing you to stop and resume an optimization task part-way through. Although it's possible to do this for every rule generated, the speed at which optimization happens would make this too inefficient. Therefore a default is set to save every 1000 rules. If you wish to tune this up/down you can do this using the `--save-every` flag.
2. Optimizing is a slow process that increases over time. The initial 10-1000 rules will be significantly slower.

### Phase 3 — Finalizing
