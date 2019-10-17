all:
	@echo "Options are: build, run"

build: nntrainer/gui/nntrainer_ui.py

run:
	$(MAKE) build
	nntrainer

nntrainer/gui/nntrainer_ui.py: nntrainer/gui/ui/nntrainer.ui
	pyuic5  -o $@ $< --from-imports

.PHONY: build run
