from flask import Flask, render_template, jsonify, request

app = Flask("nntrainer")

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/architectures")
def api_architectures():
    return jsonify(["ResNet50", "VGG16"])

@app.route("/api/upload", methods=["POST"])
def api_upload_file():
    zipfile = request.files["file"]
    print(zipfile)
    return "ok"
