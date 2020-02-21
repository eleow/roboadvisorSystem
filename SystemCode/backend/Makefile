.PHONY: clean run migrate refresh

TEST_PATH=./
scriptFile=ed_algo.py

help:
	@echo "    download"
	@echo "    	   Download data from yahoo finance and ingest 
	@echo "    run"
	@echo "    	   Run backtest for algorithm
	@echo "    clean"
	@echo "    	   Clean up zipline ingestions for robo-advisor_US


# clean:
# 	find . -name '*.pyc' -exec rm -f {} +
# 	find . -name '*.pyo' -exec rm -f {} +
# 	find . -name '*~' -exec rm -f  {} +
# 	rm -rf build/
# 	rm -rf dist/
# 	rm -rf *.egg-info
# 	rm -rf docs/_build

download:
	python $(scriptFile) -m 0 

run:
	python $(scriptFile) -m 1

clean:
	zipline ingest -b robo-advisor_US
	zipline clean -b robo-advisor_US  --keep-last 1

