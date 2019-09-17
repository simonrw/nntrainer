all: nntrainer_gui/nntrainer_ui.py

nntrainer_gui/nntrainer_ui.py: ui/nntrainer.ui Makefile
	pyuic5  -o $@ $< --from-imports
	# gsed -i 's/nnimageview/.nnimageview/' $@
	# black $@
