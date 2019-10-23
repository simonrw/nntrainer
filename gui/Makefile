all:
	@echo "Options are: build, run debug"

build: nntrainer/gui/nntrainer_ui.py nntrainer/gui/nntrainer_augmentation.py

run:
	$(MAKE) build
	nntrainer --training-dir exampledata/train --output-dir exampledata/out --name foo --validation-dir exampledata/val

debug:
	$(MAKE) build
	python -m pudb $(shell which nntrainer)

nntrainer/gui/nntrainer_ui.py: nntrainer/gui/ui/nntrainer.ui
	pyuic5  -o $@ $< --from-imports

nntrainer/gui/nntrainer_augmentation.py: nntrainer/gui/ui/nntrainer_augmentation.ui
	pyuic5  -o $@ $< --from-imports

.PHONY: build run debug
