.PHONY: clean
clean:
	rm -f models/*
	rm -f objects/*

.PHONY: lstm
lstm: lstm_model.py
	optirun ./lstm_model.py

.PHONY: cnn
cnn: cnn_model.py
	optirun ./cnn_model.py
