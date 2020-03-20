.PHONY: clean run migrate refresh

TEST_PATH=./

help:
	@echo "    front"
	@echo "    	   run frontend
	@echo "    back-run"
	@echo "    	   run backend
	@echo "    back-download"
	@echo "    	   download backend data


clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf docs/_build

front:
	make -C SystemCode/frontend/smartportfolioWeb/src run

front-refresh:
	make -C SystemCode/frontend/smartportfolioWeb/src refresh

front-migrate:
	make -C SystemCode/frontend/smartportfolioWeb/src migrate

back-run:
	make -C SystemCode/backend/ run

back-download:
	make -C SystemCode/backend/ download
