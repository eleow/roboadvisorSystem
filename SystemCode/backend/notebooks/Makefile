.PHONY: colab

help:
	@echo "    colab
	@echo "        start jupyter notebook that will allow colab frontend to connect
	@echo "        note: you should install jupyter_http_over_ws first by running install-colab
	@echo "    install_colab
	@echo "        installs jupyter_http_over_ws and enable it
	@echo "        see https://research.google.com/colaboratory/local-runtimes.html for more details
	@echo "    install_extensions
	@echo "        installs jupyter nb extensions which enables additional functionality in jupyter notebook
	@echo "        see https://towardsdatascience.com/jupyter-notebook-extensions-517fa69d2231

install_colab:
	pip install jupyter_http_over_ws
	jupyter serverextension enable --py jupyter_http_over_ws

colab:
	jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=3

install_extensions:
	pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install

