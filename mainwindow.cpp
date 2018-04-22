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

    ui->graphicsView_in->setScene(inScene);

    inPixmap = new QPixmap;
    inPixmapItem = inScene->addPixmap(*inPixmap);

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
}

void MainWindow::setFileName(const QString &fileName)
{
    this->fileName = fileName;
}

void MainWindow::setSaveFileName(const QString &saveFileName)
{
    this->saveFileName = saveFileName;
}


// rgb to gray scale, adapt from qt's function
int MainWindow::rgbToGray(const int &r, const int &g, const int &b)
{
    return (r * 11 + g * 16 + b * 5)/32;
}

void MainWindow::fisherTrain(const mat &data, const uvec &label, vec &weight, vec &dataProj, double &threshold)
{
    // 根据 label 获取类别 1 和类别 2 的数据，label 取值为 0 的为类别 1，取值为 1 的为类别 2
    // find 函数返回满足逻辑表达式的 ind
    uvec inds1 = arma::find(label == 0);
    uvec inds2 = arma::find(label == 1);
    // 注意到 data 的 size 为 Nx2 的矩阵
    // 对应着 label，应该取出 find 返回的 ind 对应的行
    mat data1 = data.rows(inds1);
    mat data2 = data.rows(inds2);
    // 本来想这样子写简单一点的，但是实际上后面还是会用到 find 函数
    // 索性全部改掉了
//    // 这里写得简单一点，类别 1 和类别 2 各占一半
//    // 并且就是输入数据的前一半为类别1，后一半为类别2
//    mat data1 = data.rows(0, data.n_rows/2 - 1);
//    mat data2 = data.rows(data.n_rows/2, data.n_rows - 1);

    // 计算均值
    mat mu1 = arma::mean(data1);
    mat mu2 = arma::mean(data2);
    // 计算散度矩阵，实际上与协方差矩阵只相差一个系数
    // 这里直接用协方差矩阵进行计算
    mat S1 = cov(data1);
    mat S2 = cov(data2);
    // 计算变换矩阵
    weight = inv(S1 + S2)*(mu1 - mu2).t();
    // 计算投影后的data
    dataProj = data*weight;

    // 下面利用投影后的 data 进行计算阈值 threshold
    // 首先求出投影后的两类数据
    vec dataProj1 = dataProj.rows(inds1);
    vec dataProj2 = dataProj.rows(inds2);
    // 以及两类数据的数量
    uword n1 = dataProj1.n_rows;
    uword n2 = dataProj2.n_rows;
    // 计算均值
    mu1 = arma::mean(dataProj1);
    mu2 = arma::mean(dataProj2);
    // 散度矩阵
    S1 = cov(dataProj1);
    S2 = cov(dataProj2);
    // 这部分还不知道什么意思，照抄的
    double a = S2(0) - S1(0);
    double b = -2*(mu1(0)*S2(0) - mu2(0)*S1(0));
    double c = S2(0)*mu1(0)*mu1(0) - S1(0)*mu2(0)*mu2(0) - 2*S1(0)*S2(0)*log(double(n1)/n2);

    double delta = b*b - 4*a*c;

    if (delta < 0) {
        QMessageBox::critical(this, tr("Error!"), tr("Something is wrong"));
        return;
    } else {
        // 利用求根公式解一元二次方程
        double d = sqrt(delta);
        double t1 = (-b + d)/(2*a);
        double t2 = (-b - d)/(2*a);

        if (pow(t1 - mu1(0), 2) < pow(t2 - mu1(0), 2)) {
            threshold = t1;
        } else {
            threshold = t2;
        }
    }
}

void MainWindow::fisherTesting(const mat &data, const vec &weight, const double &threshold, const uvec &label,
                               uvec &predictedLabel, double &precision, double &recall, double &accuracy, double &F1)
{
    vec dataProj = data*weight;
    predictedLabel = dataProj < threshold;

    // true positive, true negative
    // fasle positive, false negative


    double tp = sum(label == 1 && predictedLabel == 1);
    double tn = sum(label == 0 && predictedLabel == 0);
    double fp = sum(label == 0 && predictedLabel == 1);
    double fn = sum(label == 1 && predictedLabel == 0);

    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    accuracy = (tp + tn) / (tp + tn + fp + fn);
    F1 = 2 * precision * recall / (precision + recall);
}

// 读取 csv 文件
// 将结果按行优先顺序存储到 data 中
void MainWindow::readCsv(const QString &fileName, QVector<double> &data)
{
    QFile file(fileName);
    file.open(QIODevice::ReadOnly);
    // 初始化一个 QTextStream 流
    QTextStream stream(&file);

    while (!stream.atEnd()) {
        // 将每行读取并存入到一个字符串中
        QString str = stream.readLine();
        // 将字符串按照分隔符 ',' 分割为一个字符串列表
        QStringList list = str.split(',');

        for (int i = 0; i < list.length(); ++i) {
            data.append(list.at(i).toDouble());
        }
    }
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
}

// save output image as
void MainWindow::on_actionSave_as_triggered()
{
    QString savePath = QFileDialog::getSaveFileName(this, tr("Save image"), QDir::homePath(), imageFormat);
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
    // pad image, and then convert to grayscale
    CImg<int> inputImage = rgbToGray(padImage(img));

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
    for (uword i = 0; i < n.n_elem/2; ++i) {
        T tmp = n(i);
        n(i) = n(n.n_elem - i - 1);
        n(n.n_elem - i - 1) = tmp;
    }
    Col<T> r(n.n_elem);
    for (uword i = 0; i < n.n_elem; ++i) {
        r(i) = pow(2, i);
    }
    // polyfit
    vec x(r.n_elem);
    for (uword i = 0; i < x.n_elem; ++i) {
        x(i) = -log(r(i));
    }

    vec y(n.n_elem);
    for (uword i = 0; i < x.n_elem; ++i) {
        y(i) = log(n(i));
    }

    return mean(diff(y)/diff(x));
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

void MainWindow::on_actionFisher_triggered()
{
    mat data;
    QString dataFile = "iris.txt";
    data.load(dataFile.toStdString().data());

    mat features = data.cols(0, 1);
    vec tmp = data.col(2);
    uvec label(tmp.size());

    for (uword i = 0; i < tmp.size(); ++i) {
        label(i) = tmp(i);
    }

    // 存储输出结果
    vec weight(features.n_cols);
    vec dataProj(label.size());
    double threshold;

    fisherTrain(features, label, weight, dataProj, threshold);

    // 存储测试结果
    uvec predictedLabel(label.size());
    double precision, recall, accuracy, F1;
    fisherTesting(features, weight, threshold, label, predictedLabel, precision, recall, accuracy, F1);

    // 结果绘图
    //
    // 类别1，显然这里要用散点图
    QScatterSeries *group1 = new QScatterSeries;
    // 添加数据到 series
    for (uword i = 0; i < 50; i++) {
        group1->append(features(i, 0), features(i, 1));
    }
    // 设置名称，在图例中显示
    group1->setName(tr("Iris Setosa"));
    // 设置 marker 为 10，默认为 15
    group1->setMarkerSize(10);

    // 类别二
    QScatterSeries *group2 = new QScatterSeries;
    for (uword i = 50; i < 100; i++) {
        group2->append(features(i, 0), features(i, 1));
    }
    group2->setName(tr("Iris Versicolour"));
    // 设置 MarkerShape 为矩形，这样方便区分，group1 使用默认的圆形
    group2->setMarkerShape(QScatterSeries::MarkerShapeRectangle);
    group2->setMarkerSize(10);

    // 绘图的范围
    // 取特征点的最值在往外增加一个 offset
    double offset = 0.3;
    double xMin = floor(features.col(0).min()) - offset;
    double xMax = ceil(features.col(0).max()) + offset;
    double yMin = floor(features.col(1).min()) - offset;
    double yMax = ceil(features.col(1).max()) + offset;

    // 分类界线
    QLineSeries *border = new QLineSeries;
    for (double x = xMin; x < 2*xMax; ++x) {
        border->append(x, -weight(0)/weight(1)*x + threshold/weight(1));
    }
    border->setName(tr("Class Border"));
    // 线型设置为虚线
    border->setPen(Qt::DashLine);

    // 新建一个 QChart 对象，并将各个图添加上去
    QChart *chart = new QChart;
    chart->addSeries(group1);
    chart->addSeries(group2);
    chart->addSeries(border);

    // 创建默认的坐标轴
    chart->createDefaultAxes();
    // 设置坐标轴的范围
    chart->axisX()->setRange(xMin, xMax);
    chart->axisY()->setRange(yMin, yMax);
    // 设置一个主题
    chart->setTheme(QChart::ChartThemeBlueIcy);
    chart->setTitle(tr("Fisher Linear Discriminant Analysis"));
    // 启用动画
    chart->setAnimationOptions(QChart::AllAnimations);
    // 这个可以设置动画的时长，默认好像是 1000
    // chart->setAnimationDuration(3000);
    // 图例放在下方，图例中的 MakerShape 直接取自图表
    chart->legend()->setAlignment(Qt::AlignBottom);
    chart->legend()->setMarkerShape(QLegend::MarkerShapeFromSeries);

    // **关键**，将 chart 跟 chartView 联系起来
    ui->chartView->setChart(chart);
    // 启用抗锯齿，提升显示效果
    ui->chartView->setRenderHint(QPainter::Antialiasing);
}

// convert arma::Col to QVector
template<typename T>
Col<T> MainWindow::QVectorToCol(const QVector<T> &vector)
{
    Col<T> result(vector.length());

    for (int i = 0; i < vector.length(); ++i) {
        result(i) = vector.at(i);
    }

    return result;
}
