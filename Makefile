.PHONY: clean
clean:
	rm -f models/*
	rm -f objects/*

.PHONY: run
run: ./run.sh
	./run.sh
