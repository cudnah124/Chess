#Chess RL:

#Train/RL Bot vs Random/RL Bot vs Minimax: 

- src/train/rl_train.ipynb: 
#Cell 11: Benchmark Random Bot
#Cell 12: Benchmark Minimax Bot

#Play Human vs Bot:

```bash
python src/play.py
```

# Chess

This project focuses on processing and analyzing large-scale chess game data, with an emphasis on applying machine learning and deep learning techniques.

## Data Source

- Games are sourced from the [FICS Games Database](https://www.ficsgames.org/download.html) in PGN format.
- Data is pre-processed: invalid games are removed, move formats are standardized (PGN/FEN), and metadata (player ratings, results, openings) is extracted for downstream tasks such as classification and move prediction.

## References

- [Learning to Play Chess via Deep Reinforcement Learning](https://arxiv.org/pdf/1712.01815)
- [Deep Learning for Chess](https://erikbern.com/2014/11/29/deep-learning-for-chess)
- [Wikimedia Commons - PNG Chess Pieces](https://commons.wikimedia.org/wiki/Category:PNG_chess_pieces/Standard_transparent)

## Main Features

- Bulk PGN import and cleaning.
- Parsing and extraction of moves, board states, and player metadata.
- Generation of ML-ready datasets.
- Basic statistical analysis of games, such as opening frequency and rating distribution.

## Future Work

- Develop neural network models for move prediction and evaluation.
- Implement a full training pipeline for large datasets.

## Getting Started

1. Download data from the [FICS Games Database](https://www.ficsgames.org/download.html).
2. Run scripts in `src/preprocessing/` for cleaning and `src/features/` for feature extraction.

---

Contact [cudnah124](https://github.com/cudnah124) for questions or suggestions.