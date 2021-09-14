# Extract, Integrate, Compete: Towards Verification Style Reading Comprehension
Data and code for 'Extract, Integrate, Compete: Towards Verification Style Reading Comprehension', Findings of EMNLP 2021

Arxiv Preprint: https://arxiv.org/abs/2109.05149

## VGaokao Dataset
The VGaokao dataset is in `data/raw`
### Size

|                        | Train | Test |
|------------------------|---------------|-------------|
| Number of Passages  | 2,229          | 557         |
| Number of Questions       | 2,812          | 700         |

### Format
The format of VGaokao dataset is as follows.
```
{
  "version": "VGaokao-test",  		// dataset version
  "data": [
    {
      "cid": 3,                     // passage id
      "context": "诸子之学，兴起于先秦，当时一...", // passage
      "qas": [
        {
          "qid": "6",               // question id
          "question": "下列...不正确的一项是",   // question
          "options": [              // four options
            "广义上的...",
            "“照着讲...",
            "“接着讲...",
            "不同于以..."
          ],
          "answer": "D",            // answer  
          "correctness": [          // correctness of each option
            1,
            1,
            1,
            0
          ]
        },
        {
          ...                       // another question
        }
      ]
    }
  ]
}
```

## Extract-Integrate-Compete Method

### Dependency
* torch >= 1.8.1
* transformers == 3.5.1
* jieba >= 0.42.1

### How to Run
**Step 1: Download the resources** 
https://drive.google.com/file/d/1HbtbT4Y8YEvNqtG5aaVB_rzIdCv44R_E/view?usp=sharing
Put the `word_vectors` folder of the zip file into `data/external`.
Put the `roberta-large_ocnli` folder of the zip file into `models`.

**Step 2: Run the script**
```
cd src
./run.sh
```
The prediction result is in `models/vgaokao_soft-masking/test_mc_results.json`.
