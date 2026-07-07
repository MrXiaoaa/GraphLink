# Baseline and Benchmark Citations

This file records the third-party citations used by the GraphLink experiment settings. The main GraphLink paper citation should be added separately after publication.

The provided `main.bbl` attachment was checked first, but it contains no `\bibitem` entries. The entries below were reconstructed from the paper draft citation keys and verified against public paper records.

ReFoRCE is cited with both its paper entry and its official GitHub repository because this release includes small Spider2.0-compatible adapters derived from ReFoRCE-style utilities.

## Citation Map

| Component | Citation key | Notes |
|---|---|---|
| Spider | `yu2018spider` | Benchmark used in schema linking and SQL-generation evaluation. |
| BIRD | `li2023bird` | Benchmark used in schema linking and SQL-generation evaluation. |
| Spider2.0 / Spider2.0-Lite | `lei2024spider2` | Large-schema benchmark and backend execution setting. |
| DE-SL | `karpukhin2020dpr` | Dense dual-encoder retrieval baseline. |
| CE-SL | `khattab2020colbert` | Interaction-based reranking baseline. |
| MCS-SQL | `lee2024mcssql` | End-to-end Text-to-SQL baseline with multi-prompt selection. |
| SQL-to-Schema | `yang2024sqltoschema` | Schema-linking baseline derived from SQL drafts. |
| CHESS | `talaei2024chess` | Fixed SQL-generator and native schema-linking baseline. |
| KaSLA | `yuan2025kasla` | Knapsack-based schema-linking baseline. |
| ReFoRCE | `deng2025reforce`, `snowflakelabs2025reforcecode` | Spider2.0-compatible execution/prompt baseline referenced by compatibility utilities. Official code: https://github.com/Snowflake-Labs/ReFoRCE. |
| LinkAlign | `wang2025linkalign` | Large-schema schema-linking baseline. |
| AutoLink | `wang2025autolink` | Large-schema schema-linking baseline; `AutoLinkSL` denotes its schema-linking output in our experiment settings. |

## BibTeX

```bibtex
@inproceedings{yu2018spider,
  title = {Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author = {Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and Zhang, Zilin and Radev, Dragomir},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  year = {2018}
}

@article{li2023bird,
  title = {Can LLM Already Serve as A Database Interface? A Big Bench for Large-Scale Database Grounded Text-to-SQLs},
  author = {Li, Jinyang and Hui, Binyuan and Qu, Ge and Yang, Jiaxi and Li, Binhua and Li, Bowen and Wang, Bailin and Qin, Bowen and Cao, Rongyu and Geng, Ruiying and Huo, Nan and Zhou, Xuanhe and Ma, Chenhao and Li, Guoliang and Chang, Kevin C. C. and Huang, Fei and Cheng, Reynold and Li, Yongbin},
  journal = {arXiv preprint arXiv:2305.03111},
  year = {2023}
}

@article{lei2024spider2,
  title = {Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows},
  author = {Lei, Fangyu and Chen, Jixuan and Ye, Yuxiao and Cao, Ruisheng and Shin, Dongchan and Su, Hongjin and Suo, Zhaoqing and Gao, Hongcheng and Hu, Wenjing and Yin, Pengcheng and Zhong, Victor and Xiong, Caiming and Sun, Ruoxi and Liu, Qian and Wang, Sida and Yu, Tao},
  journal = {arXiv preprint arXiv:2411.07763},
  year = {2024}
}

@inproceedings{karpukhin2020dpr,
  title = {Dense Passage Retrieval for Open-Domain Question Answering},
  author = {Karpukhin, Vladimir and Oguz, Barlas and Min, Sewon and Lewis, Patrick and Wu, Ledell and Edunov, Sergey and Chen, Danqi and Yih, Wen-tau},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
  year = {2020}
}

@inproceedings{khattab2020colbert,
  title = {ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT},
  author = {Khattab, Omar and Zaharia, Matei},
  booktitle = {Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {2020}
}

@article{lee2024mcssql,
  title = {MCS-SQL: Leveraging Multiple Prompts and Multiple-Choice Selection For Text-to-SQL Generation},
  author = {Lee, Dongjun and Park, Choongwon and Kim, Jaehyuk and Park, Heesoo},
  journal = {arXiv preprint arXiv:2405.07467},
  year = {2024}
}

@article{yang2024sqltoschema,
  title = {SQL-to-Schema Enhances Schema Linking in Text-to-SQL},
  author = {Yang, Sun and Su, Qiong and Li, Zhishuai and Li, Ziyue and Mao, Hangyu and Liu, Chenxi and Zhao, Rui},
  journal = {arXiv preprint arXiv:2405.09593},
  year = {2024}
}

@article{talaei2024chess,
  title = {CHESS: Contextual Harnessing for Efficient SQL Synthesis},
  author = {Talaei, Shayan and Pourreza, Mohammadreza and Chang, Yu-Chen and Mirhoseini, Azalia and Saberi, Amin},
  journal = {arXiv preprint arXiv:2405.16755},
  year = {2024}
}

@article{yuan2025kasla,
  title = {Knapsack Optimization-based Schema Linking for LLM-based Text-to-SQL Generation},
  author = {Yuan, Zheng and Chen, Hao and Hong, Zijin and Zhang, Qinggang and Huang, Feiran and Li, Qing and Huang, Xiao},
  journal = {arXiv preprint arXiv:2502.12911},
  year = {2025}
}

@article{deng2025reforce,
  title = {ReFoRCE: A Text-to-SQL Agent with Self-Refinement, Consensus Enforcement, and Column Exploration},
  author = {Deng, Minghang and Ramachandran, Ashwin and Xu, Canwen and Hu, Lanxiang and Yao, Zhewei and Datta, Anupam and Zhang, Hao},
  journal = {arXiv preprint arXiv:2502.00675},
  year = {2025}
}

@misc{snowflakelabs2025reforcecode,
  title = {ReFoRCE},
  author = {{Snowflake-Labs}},
  year = {2025},
  howpublished = {\url{https://github.com/Snowflake-Labs/ReFoRCE}},
  note = {GitHub repository}
}

@article{wang2025linkalign,
  title = {LinkAlign: Scalable Schema Linking for Real-World Large-Scale Multi-Database Text-to-SQL},
  author = {Wang, Yihan and Liu, Peiyu and Yang, Xin},
  journal = {arXiv preprint arXiv:2503.18596},
  year = {2025}
}

@article{wang2025autolink,
  title = {AutoLink: Autonomous Schema Exploration and Expansion for Scalable Schema Linking in Text-to-SQL at Scale},
  author = {Wang, Ziyang and Zheng, Yuanlei and Cao, Zhenbiao and Zhang, Xiaojin and Wei, Zhongyu and Fu, Pei and Luo, Zhenbo and Chen, Wei and Bai, Xiang},
  journal = {arXiv preprint arXiv:2511.17190},
  year = {2025}
}
```
