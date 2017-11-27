.PHONY: clean
clean:
	rm -f models/*
	rm -f objects/*

.PHONY: lstm-clean
lstm-clean:
	rm -f models/lstm-*
	rm -f objects/lstm-*

.PHONY: cnn-clean
cnn-clean:
	rm -f models/cnn-*
	rm -f objects/cnn-*

.PHONY: lstm
lstm: lstm_model.py
	optirun ./lstm_model.py

.PHONY: cnn
cnn: cnn_model.py
	optirun ./cnn_model.py
