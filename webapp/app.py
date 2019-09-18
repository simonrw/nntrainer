from flask import Flask, render_template, jsonify

app = Flask("nntrainer")

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/architectures")
def api_architectures():
    return jsonify(["ResNet50", "VGG16"])
