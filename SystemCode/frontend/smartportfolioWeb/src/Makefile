.PHONY: clean run migrate refresh

TEST_PATH=./

help:
	@echo "    run"
	@echo "    	   Runs the django server
	@echo "    migrate"
	@echo "    	   Migrate the django server
	@echo "    refresh"
	@echo "    	   Collect static and clear cache


clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf docs/_build

run:
	python manage.py runserver

migrate:
	python manage.py migrate

refresh:
	python manage.py collectstatic --noinput --clear
