# Fake Image Detection - ConvNeXt Experiments

Three-class image classification with **PyTorch and ConvNeXt**, covering dataset preparation, transfer learning, five recorded experiments, and local image inference.

The model distinguishes **real**, **fully synthetic**, and **tampered** images. This repository contains the research notebooks and reference inference code. The related Streamlit/FastAPI application is described below from the project report; its frontend and backend source are not part of this upload.

## ما الذي يقدمه المشروع؟

تصنيف الصورة كاملة إلى حقيقية `real`، مولدة بالكامل `full_synthetic`، أو معدلة `tampered`، باستخدام التعلم بالنقل مع ConvNeXt. تتضمن الدفاتر تجهيز البيانات، التدريب، حفظ أفضل نموذج، منحنيات التعلم، والتقييم والتنبؤ على صور خارجية بحسب كل تجربة. لا ينتج النموذج قناعاً يحدد مكان التعديل.

تم تلخيص التقريرين الموجودين لدى صاحب المشروع داخل هذا الملف؛ ملفات PDF وأوزان النموذج غير مرفوعة إلى المستودع.

## Repository contents

| File | Purpose |
| --- | --- |
| [predict.py](predict.py) | Reference inference on images in a local `images/` directory |
| [requirements.txt](requirements.txt) | Inference dependencies: PyTorch, torchvision, Pillow |
| [requirements-notebooks.txt](requirements-notebooks.txt) | Training, evaluation, plotting, dataset and Jupyter dependencies |
| [model1.ipynb](model1.ipynb) | Experiment 1: ConvNeXt-Base, small dataset |
| [model1 (1).ipynb](model1%20%281%29.ipynb) | Experiment 2: ConvNeXt-Small, small dataset |
| [cool2 (2).ipynb](cool2%20%282%29.ipynb) | Experiment 3: ConvNeXt-Small, 50,000 images |
| [cool2 (5).ipynb](cool2%20%285%29.ipynb) | Experiment 4: ConvNeXt-Base, 50,000 images |
| [notebook88015de9df (4).ipynb](notebook88015de9df%20%284%29.ipynb) | Experiment 5: ConvNeXt-Base, 121,000 images, test evaluation and external-image predictions |

The `(4)` notebook is the later copy of the fifth experiment: it retains the same training run as `(3)` and adds two external-image inference examples. The earlier copy is omitted to keep five experiments.

## Recorded experiments

These are results already saved in the notebooks and summarized in the Arabic experiment report. They have not been reproduced by retraining during repository preparation.

| Experiment | Architecture | Dataset size | Best recorded validation accuracy |
| --- | --- | ---: | ---: |
| 1 | ConvNeXt-Base | 3,976 | 89.26% |
| 2 | ConvNeXt-Small | 3,976 | 85.40% |
| 3 | ConvNeXt-Small | 50,000 | 91.08% |
| 4 | ConvNeXt-Base | 50,000 | 91.12% |
| 5 | ConvNeXt-Base | 121,000 | **94.45%** |

The experiment report records **94.24% test accuracy** for the largest experiment on **18,150 test images**. The first four experiments do not have a saved final test result in the report. Validation and test accuracy are different measures and should not be interchanged.

### Dataset preparation

The larger notebooks reference the Hugging Face dataset identifier `saberzl/SID_Set`. Images are converted to RGB, resized and stored in class folders, with labels and split metadata. The notebooks use a seed of 42 and aim for roughly balanced classes.

| Dataset | Training | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Small, as documented in the report | 2,783 | 596 | 597 | 3,976 |
| Medium | 35,000 | 7,500 | 7,500 | 50,000 |
| Large | 84,700 | 18,150 | 18,150 | 121,000 |

The small-dataset preparation and actual image counts are not fully recoverable from its two notebooks. The large dataset retains a directory name containing `50000` in parts of the code even though the recorded total is 121,000. The held-out splits are created within the experiments; they are not an independent official benchmark split.

### Training workflow

- Start from ImageNet pretrained ConvNeXt weights and replace the final classifier with three outputs.
- Freeze features and unfreeze later layers for partial fine-tuning, according to each notebook's executed configuration.
- Train with cross-entropy, AdamW, learning-rate scheduling, and mixed precision where available.
- Track training/validation loss and accuracy; save checkpoints including model state, class names, and training state.
- Compare saved curves, evaluate the largest run, and inspect predictions on external images.

The 50,000-image experiments use 224 x 224 inputs. The largest notebook contains both 450 x 450 and 350 x 350 definitions; execution order matters. The reference inference script uses **350 x 350**. Some notebooks retain earlier definitions and inconsistent freeze-state output, so a comparison of architecture alone would overstate what these runs prove.

### Interpreting the largest test result

The report identifies a class-name ordering mismatch in the displayed test classification report. Using the confusion matrix and corrected class labels, it gives the following recalculated values:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| real | 0.9207 | 0.9083 | 0.9145 | 6,050 |
| full_synthetic | 0.9954 | 0.9983 | 0.9969 | 6,050 |
| tampered | 0.9109 | 0.9205 | 0.9157 | 6,050 |

These are report calculations from the saved matrix, not fresh predictions. The main error pattern is confusion between real and tampered images. External-image examples are qualitative checks with no independently verified labels; confidence values are not additional accuracy measurements.

## Run local inference

Python 3.10 or newer is a practical starting point. Install a matching PyTorch/torchvision pair for your CPU or CUDA environment. Dependency versions in the original project are unpinned.

```bash
git clone https://github.com/mike-elio/image-detection-fake.git
cd image-detection-fake
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Then install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Supply your checkpoint locally

**Weights are intentionally excluded.** A fresh clone requires a compatible trusted checkpoint before predictions can run; no model download is configured.

Place a checkpoint named `best_accuracy_model.pth` in the repository root or a subdirectory. The script searches recursively from the current working directory and uses the first match, so keep only the intended checkpoint in that tree.

The checkpoint must contain:

- `model_state_dict`: weights compatible with `torchvision.models.convnext_base` and its replacement classifier.
- `classes`: class names in the exact output-index order used during training. The recorded reference order is `full_synthetic`, `real`, `tampered`; the script reads the order from the checkpoint.

If your local handoff is an extracted `best_accuracy_model/` directory containing `data.pkl`, `data/`, `version` and `byteorder`, it is not itself a loadable `.pth` file. Reassemble its members under one archive prefix without changing tensor data:

```bash
python -c "from pathlib import Path; from zipfile import ZipFile, ZIP_STORED; src=Path('best_accuracy_model'); z=ZipFile('best_accuracy_model.pth','w',compression=ZIP_STORED); [z.write(p, str(Path('best_accuracy_model') / p.relative_to(src))) for p in sorted(src.rglob('*')) if p.is_file()]; z.close()"
```

This command packages an existing extracted checkpoint; it does not train or download a model. The supplied inference code uses `torch.load(..., weights_only=False)`, so only load checkpoints from a trusted source.

### Add images and predict

Create an `images/` directory and add your own JPG, JPEG, PNG, WEBP or BMP files, then run from the repository root:

```bash
python predict.py
```

The script converts each image to RGB, resizes it to 350 x 350 using bilinear resampling, applies `ToTensor` and ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), then runs ConvNeXt-Base in evaluation/inference mode. CUDA is selected when available, otherwise CPU.

Each image produces a console line with its filename, predicted class and winning softmax score as a percentage. A failed image prints an error and processing continues. Only files directly inside `images/` are processed; nested folders are not scanned. GIF is not included in this script's extension list.

The score belongs to the winning class. It is not a calibrated probability that the result is correct.

## Explore or rerun the notebooks

```bash
python -m pip install -r requirements-notebooks.txt
python -m jupyter lab
```

Open one of the five notebooks and inspect its saved outputs before running it. They were developed in Kaggle-style environments and contain `/kaggle/working` paths, dataset preparation cells and repeated configuration cells. Adjust paths, dataset access, GPU memory settings, batch sizes and worker counts for your environment. Run deliberately from a clean kernel after reviewing cell order; do not assume every notebook is a portable one-click pipeline.

The dataset, prepared image folders and checkpoints are not included. Dataset preparation needs network access and training needs substantial compute and storage. Inference requirements alone do not install the notebook toolchain.

## Related application: report summary

The separate technical report describes a **completed full-stack prototype** built around this reference model. This section documents that reported application; the Streamlit/FastAPI implementation and its test suites are not included among the local source files uploaded here.

### User journey and architecture

1. A user uploads an image in a Streamlit interface and sees a preview.
2. The frontend validates the upload and sends image bytes to a FastAPI backend.
3. The backend independently validates the input, loads the ConvNeXt checkpoint lazily, and reuses the model within the process.
4. The backend applies the same RGB conversion, 350 x 350 resize and normalization as `predict.py`.
5. The predicted three-class label is mapped to a binary verdict and returned with confidence.
6. The frontend displays the verdict or an explicit service/analysis error.

```text
Image upload -> Streamlit -> POST /analyze -> validation -> ConvNeXt-Base
                         <- verdict + confidence <- class mapping
```

The report describes two processes and no database, authentication, analysis history or persistent image storage. It also describes an explicit mock/demo analyzer mode; an API failure does not silently switch to that mode.

### Reported API contract

| Endpoint | Contract |
| --- | --- |
| `GET /health` | Returns `{"status": "ok"}` for liveness; does not establish model readiness |
| `POST /analyze` | Accepts `multipart/form-data` with an image part named `file` |
| Successful analysis | Returns `{"verdict": "real" or "fake", "confidence": number from 0.0 to 1.0}` |

The report describes JPG/JPEG, PNG, WEBP, GIF and BMP uploads with a 15 MB limit, frontend validation for usability, and backend validation including full image decoding. It records error handling for invalid files (400), oversized files (413), missing form data (422), analysis failures (500), and unavailable models (503).

The application maps `full_synthetic` to `fake`, `real` to `real`, and provisionally maps `tampered` to `fake`. That last mapping is a product decision: an edited photograph is different from a fully generated scene. Confidence remains the winning three-class probability, not the sum of both classes mapped to `fake`.

### Reported verification and limits

The application report records 72 frontend tests, 118 backend default-suite tests and 32 backend integration tests passing, plus an end-to-end run with real images. These counts belong to that application's recorded verification. They were not rerun here and are not model-accuracy benchmarks.

The experiment report and application report cover different work: the former documents training and evaluation; the latter documents software integration and explicitly states that application-level model accuracy was not independently benchmarked.

## Limitations and reproducibility

- Results reflect the saved dataset splits and configurations. Robustness to new generators, compression, editing styles and new image sources needs separate evaluation.
- No leakage or near-duplicate audit is established by these saved results.
- This is whole-image classification, with no localization or segmentation training.
- The original notebooks retain configuration inconsistencies; saved outputs and executed state need to be interpreted together.
- The inference script is a console program. The application described in the report requires source files not included in this repository.
- No weights, PDF reports, private environment files or local sample images are published in this upload.

## ملخص النتائج والتطبيق

قارنت التجارب ConvNeXt-Base وConvNeXt-Small على مجموعات صغيرة ومتوسطة وكبيرة. أفضل دقة تحقق محفوظة هي **94.45%**، ويذكر تقرير التجارب دقة اختبار **94.24%** للتجربة الأكبر. توجد ملاحظة مهمة حول ترتيب أسماء الفئات في عرض تقرير الاختبار؛ لذلك يوضح الجدول أعلاه المقاييس المصححة وفق مصفوفة الالتباس.

يشرح التقرير التقني تطبيقاً بواجهة Streamlit وخادم FastAPI يستقبل الصورة ويشغّل النموذج ويعرض نتيجة حقيقية أو مزيفة مع درجة ثقة. مصدر هذا التطبيق غير موجود في الملفات الحالية؛ الموجود هنا هو كود التنبؤ ودفاتر التدريب. الأوزان وملفات PDF مستبعدة حسب طلب صاحب المشروع، ويجب توفير النموذج محلياً لتشغيل التنبؤ.
