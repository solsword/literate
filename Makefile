.PHONY: clean
clean:
	rm -f models/*
	rm -f objects/*
	rm -f .dep_cache/*

.PHONY: lstm-clean
lstm-clean:
	rm -f models/lstm-*
	rm -f objects/lstm-*
	rm -f .dep_cache/lstm-*

.PHONY: cnn-clean
cnn-clean:
	rm -f models/cnn-*
	rm -f objects/cnn-*
	rm -f .dep_cache/cnn-*

.PHONY: lstm
lstm: main.py lstm_model.py
	optirun ./main.py lstm-rated

.PHONY: cnn
cnn: main.py cnn_model.py
	optirun ./main.py cnn-rated

%: main.py
	optirun ./main.py $@
