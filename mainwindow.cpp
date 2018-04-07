#include <QDialog>
#include <QFile>
#include <QDir>
#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>

#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    inScene = new QGraphicsScene;
    outScene = new QGraphicsScene;

    ui->graphicsView_in->setScene(inScene);
    ui->graphicsView_out->setScene(outScene);

    inPixmap = new QPixmap;
    outPixmap = new QPixmap;
    inPixmapItem = inScene->addPixmap(*inPixmap);
    outPixmapItem = outScene->addPixmap(*outPixmap);

    resultFileName = "tmp.tiff";
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_action_Quit_triggered()
{
    QApplication::quit();
}

// clean image
void MainWindow::cleanImage()
{
    inScene->clear();
    outScene->clear();
}

void MainWindow::setFileName(const QString &fileName)
{
    this->fileName = fileName;
}

void MainWindow::setSaveFileName(const QString &saveFileName)
{
    this->saveFileName = saveFileName;
}

void MainWindow::updateOutScene(const QString &fileName)
{
    qDebug() << "repaint out image." << endl;
    outScene->clear();
    outPixmap->load(fileName);
    outPixmapItem = outScene->addPixmap(*outPixmap);
    outScene->setSceneRect(QRectF(outPixmap->rect()));
}

// rgb to gray scale, adapt from qt's function
int MainWindow::rgbToGray(const int &r, const int &g, const int &b)
{
    return (r * 11 + g * 16 + b * 5)/32;
}

void MainWindow::on_action_Open_triggered()
{
    QString imagePath = QFileDialog::getOpenFileName(
                this, tr("Open file"), QDir::homePath(), imageFormat);

    // check if file is valid
    if (!imagePath.isEmpty()) {
        QFile file(imagePath);

        if(!file.open(QIODevice::ReadOnly)) {
            QMessageBox::critical(this, tr("Error"), tr("Unable to read image!"));
            return;
        }

        // clear previouly showed image
        cleanImage();

        // show image
        inPixmap->load(imagePath);
        inPixmapItem = inScene->addPixmap(*inPixmap);
        inScene->setSceneRect(QRectF(inPixmap->rect()));

        // save fileName for later use
        setFileName(imagePath);
    }
}

// save output image
void MainWindow::on_action_Save_triggered()
{
    if (!saveFileName.isEmpty()) {
        outPixmapItem->pixmap().save(saveFileName);
    } else {
        on_actionSave_as_triggered();
    }
}

// save output image as
void MainWindow::on_actionSave_as_triggered()
{
    QString savePath = QFileDialog::getSaveFileName(this, tr("Save image"), QDir::homePath(), imageFormat);

    if (!savePath.isEmpty()) {
        QFile file(savePath);

        if (!file.open(QIODevice::WriteOnly)) {
            QMessageBox::critical(this, tr("Error!"), tr("Unable to save image!"));
            return;
        }

        outPixmapItem->pixmap().save(savePath);
        setSaveFileName(savePath);
    }
}

void MainWindow::on_actionClose_triggered()
{
    cleanImage();
    setFileName("");
}

void MainWindow::on_actionGLCM_triggered()
{
    dlgGLCM = new DialogGLCM;

    dlgGLCM->setModal(true);
    dlgGLCM->show();
    connect(dlgGLCM,
            SIGNAL(sendData(int, int, int)),
            this,
            SLOT(glcm(int, int, int)));
}

void MainWindow::glcm(const int &distance, const int &theta, const int &grayLevel)
{
    CImg<int> img(fileName.toStdString().data());
    CImg<int> grayImg(img.width(), img.height());
    CImg<int> glcmImg(grayLevel, grayLevel, 1, 1, 0);
    CImg<double> glcmImgNorm(glcmImg);

    // make sure input image is grayscale
    if (!isGrayscale(img)) {
        grayImg = rgbToGray(img);
    } else {
        grayImg = img;
    }
    // firstly, do histogram equalize, then quantize to grayLevel to reduce glcm matrix size
    grayImg.equalize(256).quantize(grayLevel, false);
    glcmImg = getGlcm(grayImg, distance, theta, grayLevel);
    // normalization
    glcmImgNorm = glcmImg/glcmImg.sum();
    // features needed to calculate
    // Angular Second Moment, ASM
    // Inverse Differential Moment, IDM
    // Contrast
    // Correlation
    // Entropy
    double ASM = getASM(glcmImgNorm);
    double IDM = getIDM(glcmImgNorm);
    double contrast = getContrast(glcmImgNorm);
    double correlation = getCorrelation(glcmImgNorm);
    double entropy = getEntropy(glcmImgNorm);
    // pop up a messagebox to show result
    QMessageBox resultBox;
    QString resultString = tr("ASM:\t\t%1\nIDM:\t\t%2\nContrast:\t\t%3\nCorrelation:\t%4\nEntropy:\t\t%5").arg(ASM).arg(IDM).arg(contrast).arg(correlation).arg(entropy);
    resultBox.setText(resultString);
    resultBox.setWindowTitle(tr("Texture Features"));
    resultBox.exec();
}

template <typename T>
bool MainWindow::isGrayscale(const CImg<T> &img)
{
    if (img.spectrum() == 1) {
        return true;
    } else {
        return false;
    }
}

template <typename T>
CImg<T> MainWindow::getGlcm(const CImg<T> &img, const int &distance, const int &theta, const int &grayLevel)
{
    CImg<T> glcmImg(grayLevel, grayLevel, 1, 1, 0);
    int rows, cols;

    switch(theta) {
    case 0:
        cimg_forXY(img, x, y) {
            if (x + distance < img.width()) {
                rows = img(x, y);
                cols = img(x + distance, y);
                glcmImg(rows, cols)++;
            }
        }
        break;

    case 45:
        cimg_forXY(img, x, y) {
            if (x - distance >= 0 && y + distance < img.height()) {
                rows = img(x, y);
                cols = img(x - distance, y + distance);
            }
        }
        break;

    case 90:
        cimg_forXY(img, x, y) {
            if (x - distance >= 0) {
                rows = img(x, y);
                cols = img(x - distance, y);
            }
        }
        break;

    case 135:
        cimg_forXY(img, x, y) {
            if (x - distance >= 0 && y - distance >= 0) {
                rows = img(x, y);
                cols = img(x - distance, y - distance);
            }
        }
        break;

    default:
        QMessageBox::critical(this, tr("Error!"), tr("Oops, something is wrong!"));
    }

    return glcmImg;
}

template <typename T>
double MainWindow::getEntropy(const CImg<T> &img)
{
    double result = 0.0f;

    cimg_forXY(img, x, y) {
        result += img(x, y)*log2(img(x, y) + DBL_EPSILON);
    }

    return -result;
}

template<typename T>
double MainWindow::getASM(const CImg<T> &img)
{
    double result = 0.0f;

    cimg_forXY(img, x, y) {
        result += img(x, y)*img(x, y);
    }

    return result;
}

template<typename T>
double MainWindow::getIDM(const CImg<T> &img)
{
    double result = 0.0f;

    cimg_forXY(img, x, y) {
        result += img(x, y)/(1 + (x - y)*(x - y));
    }

    return result;
}

template<typename T>
double MainWindow::getContrast(const CImg<T> &img)
{
    double result = 0.0f;

    cimg_forXY(img, x, y) {
        result += (x - y)*(x - y)*img(x, y);
    }

    return result;
}

template<typename T>
double MainWindow::getCorrelation(const CImg<T> &img)
{
    double result = 0.0f;
    CImg<T> Px(img.width(), 1, 1, 1, 0);
    CImg<T> Py(1, img.height(), 1, 1, 0);

    cimg_forXY(img, x, y) {
        Px(x, 0) += img(x, y);
        Py(0, y) += img(x, y);
    }

    double meanX = 0.0f;
    double meanY = 0.0f;
    double varianceX = 0.0f;
    double varianceY = 0.0f;

    cimg_forXY(Px, x, y) {
        meanX += x*Px(x, y);
    }

    cimg_forXY(Px, x, y) {
        varianceX += (x - meanX)*(x - meanX)*Px(x, y);
    }

    cimg_forXY(Py, x, y) {
        meanY += y*Py(x, y);
    }

    cimg_forXY(Py, x, y) {
        varianceY += (y - meanY)*(y - meanY)*Py(x, y);
    }

    cimg_forXY(img, x, y) {
        result += (x - meanX)*(y - meanY)*img(x, y)/(varianceX*varianceY + DBL_EPSILON);
    }

    return result;
}

void MainWindow::on_actionFractal_dimension_triggered()
{
    CImg<double> img(fileName.toStdString().data());
    CImg<int> inputImage = rgbToGray(img);

    // convert to binary image with threshold = 128
    double threshold = 128.0f;
    cimg_forXY(inputImage, x, y) {
        inputImage(x, y) = inputImage(x, y) > threshold ? 1 : 0;
    }

    // 计算分形维度
    double dim = boxcount(cimgToMat(inputImage));
    // pop up a messagebox to show result
    QMessageBox resultBox;
    QString resultString = tr("fractal dimension: %1").arg(dim);
    resultBox.setText(resultString);
    resultBox.setWindowTitle(tr("Texture Features"));
    resultBox.exec();
}

// convert 2D grayscale image to a matrix
template<typename T>
Mat<T> MainWindow::cimgToMat(const CImg<T> &img)
{
    Mat<T> result(img.width(), img.height());

    cimg_forXY(img, x, y) {
        result(x, y) = img(x, y);
    }

    return result;
}

// 根据 http://m2matlabdb.ma.tum.de/files.jsp?MC_ID=5&SC_ID=13 的 MATLAB 版本修改而来
// 缺点是对于自相似程度较低的图像的结果误差较大
template<typename T>
double MainWindow::boxcount(const Mat<T> &img)
{
    Mat<T> c = img;
    int width = c.n_rows;
    int p = log(width)/log(2);
    Col<T> n(p+1, fill::zeros);
    n(p) = accu(c);

    for (int g = p-1; g >= 0; --g) {
        int siz = pow(2, p-g);
        int siz2 = round(siz/2);
        for (int i = 0; i < width - siz + 1; i += siz) {
            for (int j = 0; j < width - siz + 1; j += siz) {
                c(i, j) = c(i, j) || c(i+siz2, j) || c(i, j+siz2) || c(i+siz2, j+siz2);
            }
        }
        for (int u = 0; u < width - siz + 1; u += siz) {
            for (int v = 0; v < width - siz + 1; v += siz) {
                n(g) += c(u, v);
            }
        }
    }
    // 倒序 n = n(end:-1:1)
    for (int i = 0; i < n.n_elem/2; ++i) {
        T tmp = n(i);
        n(i) = n(n.n_elem - i - 1);
        n(n.n_elem - i - 1) = tmp;
    }
    Col<T> r(n.n_elem);
    for (int i = 0; i < n.n_elem; ++i) {
        r(i) = pow(2, i);
    }
    // polyfit
    vec x(r.n_elem);
    for (int i = 0; i < x.n_elem; ++i) {
        x(i) = -log(r(i));
    }

    vec y(n.n_elem);
    for (int i = 0; i < x.n_elem; ++i) {
        y(i) = log(n(i));
    }
    vec P = polyfit(x, y, 1);

    return P(0);
}

// pad image with zeros from size MxN to M'xM',
// where M' = 2^K
template<typename T>
CImg<T> MainWindow::padImage(const CImg<T> &img)
{
    // if width == height, just scale
    if (img.width() == img.height()) {
        unsigned int M = pow(2, ceil(log2(img.width())));
        return img.get_resize(M, M, img.depth(), img.spectrum(), 5);
    } else {
        // width != height, pad with zeros
        unsigned int M = pow(2, ceil(log2(img.width() > img.height() ? img.width() : img.height())));
        return img.get_resize(M, M, img.depth(), img.spectrum(), 0);
    }
}

template<typename T>
CImg<T> MainWindow::rgbToGray(const CImg<T> &img)
{
    CImg<T> result(img.width(), img.height());

    if (img.spectrum() == 3) {
        cimg_forXY(img, x, y) {
            result(x, y) = rgbToGray(img(x, y, 0), img(x, y, 1), img(x, y, 2));
        }
    } else {
        result = img;
    }

    return result;
}
