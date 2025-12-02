# CUDARule Scoring & Optimization
What started as a hobby project in Q4 of 2024 out of interest in what optimizations were possible in rules, has now grown into a much larger project. With a lot of hardware support from the community I've built a tool that can score rules, counting the potential founds between a wordlist+rules and a target wordlist.

This tool presents a comprehensive package to score rules and optimize rulefiles, finding noise in the madness of rules. Prior to the release of this tool it was not possible to easily determine if a rulefile was effective except by running it through hashcat or RuleProcessorY, but using the tools 'score' functionality you can accurate determine how many founds a specific rule will have given an input wordlist (the same input wordlist used to attack hashcat in hashcat).

HashMob is a password recovery community that launched back in 2021 and has since gained a large number of followers and active contributors. A weekly wordlist containing over 10,000 data breaches helps provide high quality plaintexts in near- realtime. Because of the large amount of statistically aggregated data (following the law of large numbers) we highly recommend using the HashMob huge or HashMob Combined Full as your 'target' wordlist when using this tool.
This will not only ensure that your ruleset is optimized to reflect real-world data, but also that it is statistically relevant. HashMob huge for example only takes the plaintexts that occur in >2 wordlists from over 10,000 data breaches -> A statistically significant dataset. 

The program is based on CUDA SIMD programming and requires the use of NVCC to compile the kernel. Primary development happened using Cakes' hashcat docker container developed for Team HashMob with the use of Vast.AI and users utilizing windows might fight an increased difficulty in getting the program functional, I appologize for this in advance and welcome any PR with instructions to get Windows operational.

The base wordlist used for HashMob rules is the HashMob Large wordlist. This wordlist contains plaintexts that occur in >5 hashlists.

Help:
```
Usage: ruleSetOptimizer <command> [flags]

An application that optimizes Hashcat rules using set coverage optimization theory based on rule performance.

Flags:
  -h, --help               Show context-sensitive help.
  -x, --session=default    Session Name.

Commands:
  score --rule-file=best66.rule --output-file=best66.score <wordlist> <target> [flags]
    Score rule files.

  optimize --score-file=best66.score --output-file=best66.optimized <wordlist> <target> [flags]
    Optimize a score file.

  simulate --rule-file=best66.rule --output-file=best66.sim <wordlist> <target> [flags]
    Run a simulation on the target list.

  format --score-file=best66.score --output-file=best66.rule [flags]
    Remove the scores from the TSV file and transform it into a hashcat-compatible file.

  version [flags]
    Version & Author information

Run "ruleSetOptimizer <command> --help" for more information on a command.

ruleSetOptimizer: error: expected one of "score", "optimize", "simulate", "format", "version"
```


## Linux (Ubuntu) Quick-Start
The tool is built in Golang and requires Go to operate, with CGO as bridge to the CUDA kernel. To have the CUDA kernel compile the CUDA toolkit must be installed.
```bash
# Ensure your system is up to date.
apt update -y && apt dist-upgrade -y
# Install GoLang. The version below shows the latest version at time of release, feel free to take a more recent version.
wget https://go.dev/dl/go1.25.1.linux-amd64.tar.gz
rm -rf /usr/local/go && tar -C /usr/local -xzf go1.25.1.linux-amd64.tar.gz
# Install the NVIDIA Toolkit
sudo apt install -y nvidia-cuda-toolkit
```
```text
# Add the following variables to ~/.profile to ensure the required Go & NVIDIA libraries can be found
export PATH=/usr/local/nvidia/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/go/bin:$PATH
export CPATH=/usr/local/cuda/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
```bash
# Then run the following to load the environment variables from ~/.profile:
source ~/.profile

# Compile the project using the build.sh or individual commands.
chmod +x build.sh
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
2. Data Origin and quality matters
3. Character types matter. It's important to 
4. Clean your data! `rehex -u <target|wordlist>` and `rling target.txt target.txt wordlist.txt` will help ensure your data is clean, unique, and unencoded.
5. When it comes to choosing rules, more is better; but will also increase the runtime linearly.
#### Memory usage
When talking about memory usage we try to limit the amount used as much as possible, but there are no way around some requirements. This means that fitting your dataset might require a certain amount of GPU VRAM. The easiest way to calculate a close estimate is to use the following formula:
```total memory usage = wordlistCount*5 + targetCount*5```
This will not be perfect as rules will have to be loaded in as well but becomes a close estimate.

### Phase 1 — Scoring
The first phase has the goal of scoring the performance of rules as if it was the first rule ran. This helps prioritize rules over others and enables larger optimizations later. This phase can be ran on as many rules as you wish and you can cut-off or remove low-scoring entries as you desire as long as the input and target wordlist remain unadjusted.

```
Usage: ruleSetOptimizer score --rule-file=best66.rule --output-file=best66.score <wordlist> <target> [flags]

Score rule files.

Arguments:
  <wordlist>    Path to wordlist file
  <target>      Path to target data file

Flags:
  -h, --help                        Show context-sensitive help.
  -x, --session=default             Session Name.

  -r, --rule-file=best66.rule       Rule file to analyse.
  -o, --output-file=best66.score    Score File to output results to.

ruleSetOptimizer: error: missing flags: --output-file=best66.score, --rule-file=best66.rule
```

```
./ruleSetOptimizer score hashmob.medium.txt hashmob.huge.txt -r rules.txt -o output.score 
```

Example of a score file:
```
57270192        :
39572   $
2080914 $!
218213  $#
375211  $$
69374   $%
5465140 $0
9352871 $1
6455906 $2
5800259 $3
5095255 $4
5156365 $5
4636704 $6
5142937 $7
4660270 $8
4674669 $9
465     $7      $0      $1      $8      o66     *60     x38     T0
465     $*      $2      $6      *8A     'A      s1.
465     o45     i44
465     C       $3      $2

```

### Phase 2 — Optimizing
This phase will follow the scoring phase once you've scored all rules you want to evaluate and optimize. It is recommended to create a single file containing all scores and sorting them with the following command:
```shell
cat *.score | sort -u | sort -rn > all_scores.dat
```
Then run the optimize command to start optimizing. Before starting the optimize process there are a few points to be mindful of:
1. A pause and resume function is available, allowing you to stop and resume an optimization task part-way through. Although it's possible to do this for every rule generated, the speed at which optimization happens would make this too inefficient. Therefore a default is set to save every 1000 rules. If you wish to tune this up/down you can do this using the `--save-every` flag.
2. Optimizing is a slow process that increases its speed over time. The initial 10-1000 rules will be significantly slower.

```
Usage: ruleSetOptimizer optimize --score-file=best66.score --output-file=best66.optimized <wordlist> <target> [flags]

Optimize a score file.

Arguments:
  <wordlist>    Path to wordlist file
  <target>      Path to target data file

Flags:
  -h, --help                            Show context-sensitive help.
  -x, --session=default                 Session Name.

  -s, --score-file=best66.score         Aggregated score file TSV.
  -o, --output-file=best66.optimized    Score File to output results to.
      --save-every=1000                 Save progress every x rules.

ruleSetOptimizer: error: missing flags: --output-file=best66.optimized, --score-file=best66.score
```


```shell
./ruleSetOptimizer optimize hashmob.medium.txt hashmob.huge.txt -s all_scores.dat -o optimized.rule_sim 
```

### Phase 3 — Finalizing
Finally, to remove the scores and turn it into a usable rule-file we need to remove the first column of the output file and replace tabs with space.
There are a few ways to do this, use the built-in command, use linux commands, or your own editor

```
Usage: ruleSetOptimizer format --score-file=best66.score --output-file=best66.rule [flags]

Remove the scores from the TSV file and transform it into a hashcat-compatible file.

Flags:
  -h, --help                       Show context-sensitive help.
  -x, --session=default            Session Name.

  -s, --score-file=best66.score    Aggregated score file TSV.
  -o, --output-file=best66.rule    Hashcat rule file output.

ruleSetOptimizer: error: missing flags: --output-file=best66.rule, --score-file=best66.score
```
The command to format it can be either:
```bash
./ruleSetOptimizer format -s optimized.rule_sim -o optimized.rule
```
Or
```bash
perl -pe "s/^(\d+)\t(.*)$/\1/" optimized.rule_sim | perl -pe "s/\t/ /g" > optimized.rule
```

### Simulating
Simulate will emulate a hashcat rule attack on a selected 'target' (hashlist). This command will take the first rule, process it, and count the matches in the target file. Remove them from the target file, and then move to the next one where it will repeat the process until all rules are completed. This is an O(N) operation as opposed to the Optimizing process.

This feature is great for mapping the efficiency of a word- and rule- file on a specific dataset and allows you to draw some very interesting charts that highlight what rules are over/under performing, or how they compare against other files.
```
Usage: ruleSetOptimizer simulate --rule-file=best66.rule --output-file=best66.sim <wordlist> <target> [flags]

Run a simulation on the target list.

Arguments:
  <wordlist>    Path to wordlist file
  <target>      Path to target data file

Flags:
  -h, --help                      Show context-sensitive help.
  -x, --session=default           Session Name.

  -r, --rule-file=best66.rule     Rule file to analyse.
  -o, --output-file=best66.sim    Score File to output results to.
  -d, --device-id=0               Device ID.

ruleSetOptimizer: error: missing flags: --output-file=best66.sim, --rule-file=best66.rule
```

```bash
./ruleSetOptimizer simulate -s optimized.rule_sim -o optimized.rule
```