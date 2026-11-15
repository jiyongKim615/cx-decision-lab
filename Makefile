PYTHON ?= python3

.PHONY: lab test evaluate demo
lab: test
	$(PYTHON) -m compileall -q cx_research_lab scripts tests
	$(PYTHON) -m scripts.evaluate

test:
	$(PYTHON) -m unittest discover -s tests -v

evaluate:
	$(PYTHON) -m scripts.evaluate

demo:
	$(PYTHON) -m cx_research_lab
