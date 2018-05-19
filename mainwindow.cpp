#include <QDialog>
#include <QFile>
#include <QDir>
#include <QFileDialog>
#include <QMessageBox>
#include <QVector>
#include <QDebug>

#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_action_Quit_triggered()
{
    QApplication::quit();
}

void MainWindow::setFileName(const QString &fileName)
{
    this->fileName = fileName;
}

// rgb to gray scale, adapt from qt's function
int MainWindow::rgbToGray(const int &r,
                          const int &g,
                          const int &b)
{
    return (r * 11 + g * 16 + b * 5)/32;
}

// 灰度共生矩阵的计算
// 输入 SI 是经过灰度级量化之后的矩阵
// grayLevel 是 SI 的灰度级，一般取值为 8, 16, 32, 64, 128, 256 等
// rowStep 和 colStep 分别是行方向和列方向的位移
// 更多信息可以参考 https://github.com/palmerc/GLCM/blob/master/GLCM.m
mat MainWindow::graycomatrix(const Mat<int> &SI,
                             const int &grayLevel,
                             const int &rowStep,
                             const int &colStep)
{
    // 用于保存结果的变量
    // 显然大小为 grayLevel x grayLevel
    // 并且填满 0 值
    mat result(grayLevel, grayLevel, arma::fill::zeros);

    // 生成灰度共生矩阵的关键
    // 遍历整个 SI 矩阵
    // 统计灰度值对 (intensity1, intensity2)
    // 在指定的距离和方向上出现的次数
    // 指定的距离和方向由 (rowStep, colStep) 确定
    //
    // 将变量定义在循环外，避免每次循环都新建变量
    int intensity1, intensity2;
    for (uword row = 0; row < SI.n_rows; ++row) {
        for (uword col = 0; col < SI.n_cols; ++col) {
            intensity1 = SI(row, col);
            // 确认该点还在矩阵范围内
            if (row + rowStep < SI.n_rows && col + colStep < SI.n_cols) {
                intensity2 = SI(row + rowStep, col + colStep);
                result(intensity1, intensity2) += 1;
            }
        }
    }

    // 上面计算得到的灰度共生矩阵并不是对称矩阵
    // 也就是说 0 度角和 180 度角的灰度共生矩阵是不一样的
    // 但是一般常用的还是要对称矩阵
    // 要得到对称矩阵，只需将其与转置矩阵相加即可
    // 因为后续计算的时候是使用的归一化的灰度共生矩阵
    // 所以并不需要 (result + result.t()) / 2

    return result + result.t();
}

// ASM，角二阶矩，也称作能量
// 计算公式为 \sum_{i,j}^{k} p(i, j)^2
double MainWindow::getASM(const mat &glcmMatrix)
{
    double result = 0.0f;

    for (uword i = 0; i < glcmMatrix.n_rows; ++i) {
        for (uword j = 0; j < glcmMatrix.n_cols; ++j) {
            result += glcmMatrix(i, j)*glcmMatrix(i, j);
        }
    }

    return result;
}

// IDM，逆差矩
// 计算公式 \sum_{i, j}^{k} p(i, j)/(1 + (i - j)^2)
double MainWindow::getIDM(const mat &glcmMatrix)
{
    double result = 0.0f;

    for (uword i = 0; i < glcmMatrix.n_rows; ++i) {
        for (uword j = 0; j < glcmMatrix.n_cols; ++j) {
            result += glcmMatrix(i, j) / (1 + (i - j)*(i - j));
        }
    }

    return result;
}

// Contrast，对比度
// 计算公式 sum_{i, j}^{k} (i - j)^2 p(i, j)
double MainWindow::getContrast(const mat &glcmMatrix)
{
    double result = 0.0f;

    for (uword i = 0; i < glcmMatrix.n_rows; ++i) {
        for (uword j = 0; j < glcmMatrix.n_cols; ++j) {
            result += (i - j)*(i - j)*glcmMatrix(i, j);
        }
    }

    return result;
}

// Correlation，相关
// 计算公式 \sum_{i, j}^{k} p(i, j)(i - mu_r)(j - mu_c)/(sigma_r*sigma_c)
// 其中 mu_r, mu_c 为灰度共生矩阵的行和列均值
// sigma_r, sigma_c 为灰度共生矩阵的行和列方差
double MainWindow::getCorrelation(const mat &glcmMatrix)
{
    // 分别将行相加和列相加
    vec px = sum(glcmMatrix, 1);
    vec py = sum(glcmMatrix, 0).t();
    // 计算统计量均值和方差
    // 注意，这里的输入为归一化之后的灰度共生矩阵
    // 也就是说这个矩阵中的每个元素代表的是频率
    // 所以均值和方差的计算不能直接调用 mean 和 var 函数
    double mu_x = 0.0f;
    double mu_y = 0.0f;
    double sigma_x = 0.0f;
    double sigma_y = 0.0f;

    for (uword i = 0; i < px.n_elem; ++i) {
        mu_x += i*px(i);
    }

    for (uword i = 0; i < px.n_elem; ++i) {
        sigma_x += (i - mu_x)*(i - mu_x)*px(i);
    }

    for (uword i = 0; i < py.n_elem; ++i) {
        mu_y += i*py(i);
    }

    for (uword i = 0; i < py.n_elem; ++i) {
        sigma_y += (i - mu_y)*(i - mu_y)*py(i);
    }

    double result = 0.0f;
    for (uword i = 0; i < glcmMatrix.n_rows; ++i) {
        for (uword j = 0; j < glcmMatrix.n_cols; ++j) {
            result += glcmMatrix(i, j) * (i - mu_x) * (j - mu_y) / (sigma_x*sigma_y);
        }
    }

    return result;
}

// Entropy，熵
// 计算公式 -\sum_{i, j}^{k} p(i, j)log2(p(i, j))
double MainWindow::getEntropy(const mat &glcmMatrix)
{
    double result = 0.0f;

    for (uword i = 0; i < glcmMatrix.n_rows; ++i) {
        for (uword j = 0; j < glcmMatrix.n_cols; ++j) {
            // 注意这里的 log2(N) 的 N 可能为 0
            result += -glcmMatrix(i, j)*log2(glcmMatrix(i, j) + DBL_EPSILON);
        }
    }

    return result;
}

void MainWindow::fisherTrain(const mat &data,
                             const ivec &label,
                             vec &weight,
                             vec &dataProj,
                             double &threshold)
{
    // 根据 label 获取类别 1 和类别 2 的数据，label 取值为 -1 的为类别 1，取值为 1 的为类别 2
    // 这里 label 取值 -1 是因为跟感知机的保持一致，这样就可以使用相同的数据文件
    // 显然 label 取值只要能将两类区别开即可
    //
    // find 函数返回满足逻辑表达式的 ind
    uvec inds1 = arma::find(label == -1);
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
    weight = inv(S1 + S2)*(mu2 - mu1).t();
    // 计算投影后的data
    dataProj = data*weight;

    // 方法一
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
    // 计算一元二次方程的系数
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
    // 方法二，直接取投影后的均值作为阈值
    // threshold = mean(dataProj);
}

void MainWindow::fisherTest(const mat &data,
                            const vec &weight,
                            const double &threshold,
                            const ivec &label,
                            ivec &predictedLabel,
                            double &precision,
                            double &recall,
                            double &accuracy,
                            double &F1)
{
    vec dataProj = data*weight;
    // 计算预测的标签
    for (uword i = 0; i < predictedLabel.n_elem; ++i) {
        if (dataProj(i) < threshold) {
            predictedLabel(i) = -1;
        } else {
            predictedLabel(i) = 1;
        }
    }

    // true positive, true negative
    // fasle positive, false negative
    double tp = sum(label == 1 && predictedLabel == 1);
    double tn = sum(label == -1 && predictedLabel == -1);
    double fp = sum(label == -1 && predictedLabel == 1);
    double fn = sum(label == 1 && predictedLabel == -1);

    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    accuracy = (tp + tn) / (tp + tn + fp + fn);
    F1 = 2 * precision * recall / (precision + recall);
}

// 感知机算法实现
// 输入为 x 为 Nxd 的矩阵，其中 N 为样本数量，d 为特征维数
// 作业要求是 2 维的，但是这个算法应该是通用的
// y 是标签，为一个 N 维列向量，取值为 {-1, 1}
// learningRate 为学习率
// maxEpoch 为最大迭代次数
// weight 为所需要学习的权重，为 d+1 维向量
// weight 的最后一维为偏置
void MainWindow::perceptionTrain(const mat &data,
                                 const ivec &label,
                                 const double &learningRate,
                                 const int &maxEpoch,
                                 vec &weight)
{
    // 输入数据的 size
    uword m = data.n_rows;
    uword n = data.n_cols;
    // 将输入数据改写为增广形式
    mat x = ones(m, n+1);
    x.cols(0, n - 1) = data;

    // 训练结束的标志
    bool finishFlag;

    // 这个循环就是训练的主要代码
    for (int epoch = 0; epoch < maxEpoch; ++epoch) {
        finishFlag = true;

        for (uword i = 0; i < m; ++i) {
            // 这种方式直接判断 label 的，没有用到损失函数
            //            if (sign(dot(x.row(i), weight)) != label(i)) {
            //                finishFlag = false;
            //                weight += learningRate * label(i) * x.row(i).t();
            //            }
            // 这种方式才是使用损失函数来判断的
            if (label(i) * (dot(x.row(i), weight)) <= 0) {
                finishFlag = false;
                weight += learningRate * label(i) * x.row(i).t();
            }
        }

        if (finishFlag) {
            break;
        }
    }
}

// 感知机测试函数
// 输入 data, label, weight 等参数与训练函数的类似
// 其他参数与 fisher 的测试函数类似
void MainWindow::perceptionTest(const mat &data,
                                const ivec &label,
                                const vec &weight,
                                ivec &predictedLabel,
                                double &precision,
                                double &recall,
                                double &accuracy,
                                double &F1)
{
    // 输入数据的 size
    uword m = data.n_rows;
    uword n = data.n_cols;
    // 将输入数据改写为增广形式
    mat x = ones(m, n+1);
    x.cols(0, n - 1) = data;

    // 计算预测的标签
    for (uword i = 0; i < label.n_elem; ++i) {
        predictedLabel(i) = sign(dot(x.row(i), weight));
    }
    // 计算 tp, tn, fp, fn
    // 然后根据这些计算精确率准确率召回率等
    double tp = sum(label == 1 && predictedLabel == 1);
    double tn = sum(label == -1 && predictedLabel == -1);
    double fp = sum(label == -1 && predictedLabel == 1);
    double fn = sum(label == 1 && predictedLabel == -1);

    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    accuracy = (tp + tn) / (tp + tn + fp + fn);
    F1 = 2 * precision * recall / (precision + recall);
}

void MainWindow::on_action_Open_triggered()
{
    // get image file name
    QString imagePath = QFileDialog::getOpenFileName(
                this, tr("Open file"), QDir::homePath(), imageFormat);

    // if imagePath is invalid, just return, and do nothing.
    if (imagePath.isEmpty()) {
        return;
    }
    // create QPixmap object, and init with image file name
    QPixmap pixmap(imagePath);
    // create QGraphicsScene
    QGraphicsScene *scene = new QGraphicsScene;

    // add pixmap to scene
    scene->addPixmap(pixmap);
    // attach QGrapicsView with QGraphicsScene
    ui->graphicsView->setScene(scene);

    // update fileName for later use
    setFileName(imagePath);
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

void MainWindow::glcm(const int &rowStep,
                      const int &colStep,
                      const int &grayLevel)
{
    CImg<int> img(fileName.toStdString().data());
    CImg<int> grayImg(img.width(), img.height());
    CImg<int> SIImg(grayLevel, grayLevel);

    // 确认图像为灰度图，否则就转换为灰度图
    if (!isGrayscale(img)) {
        grayImg = rgbToGray(img);
    } else {
        grayImg = img;
    }
    // 先进行直方图均衡，然后重新量化，减少灰度级
    // 注意，这个量化的算法与 MATLAB 中的 graycomatrix 函数中用到的量化算法不一样
    // 所求求出来的 SI 矩阵是不同的，进而导致求出的特征也不同
    SIImg = grayImg.get_equalize(256).get_quantize(grayLevel, false);
    // 转换为 arma::Mat 类型
    Mat<int> SI = cimgToMat(SIImg);

    // 计算灰度共生矩阵
    mat glcmMat = graycomatrix(SI, grayLevel, rowStep, colStep);

    // 将灰度共生矩阵归一化
    glcmMat /= accu(glcmMat);
    Mat<double> glcmMatNorm = glcmMat/accu(glcmMat);

    // features needed to calculate
    // Angular Second Moment, ASM
    // Inverse Differential Moment, IDM
    // Contrast
    // Correlation
    // Entropy

    double ASM = getASM(glcmMatNorm);
    double IDM = getIDM(glcmMatNorm);
    double contrast = getContrast(glcmMatNorm);
    double correlation = getCorrelation(glcmMatNorm);
    double entropy = getEntropy(glcmMatNorm);

    // pop up a messagebox to show result
    QMessageBox resultBox;
    QString resultString = tr("ASM:\t\t%1\nIDM:\t\t%2\nContrast:\t%3\nCorrelation:\t%4\nEntropy:\t\t%5").arg(ASM).arg(IDM).arg(contrast).arg(correlation).arg(entropy);
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

// 将 CImg.h 的图像转换为 armadillo 的 Mat 矩阵
// 因为 Mat 是 2D 矩阵，所以输入只能是灰度图像
template <typename T>
Mat<T> MainWindow::cimgToMat(const CImg<T> &img)
{
    Mat<T> result(img.width(), img.height());

    // 直接在一个循环内将所有的像素点的灰度值赋值给对应的矩阵即可
    cimg_forXY(img, x, y) {
        result(x, y) = img(x, y);
    }

    return result;
}

// 根据 http://m2matlabdb.ma.tum.de/files.jsp?MC_ID=5&SC_ID=13 的 MATLAB 版本修改而来
// 缺点是对于自相似程度较低的图像的结果误差较大
template <typename T>
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
template <typename T>
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

template <typename T>
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
    dlgRandomData2 = new DialogRandomData2;

    dlgRandomData2->show();

    connect(dlgRandomData2,
            SIGNAL(sendData(int, vec, mat, int, vec, mat)),
            this,
            SLOT(fisher(int, vec, mat, int, vec, mat)));
}

// convert arma::Col to QVector
template <typename T>
Col<T> MainWindow::QVectorToCol(const QVector<T> &vector)
{
    Col<T> result(vector.length());

    for (int i = 0; i < vector.length(); ++i) {
        result(i) = vector.at(i);
    }

    return result;
}

// 符号函数
// x >=0 时返回 1，否则返回 -1
template <typename T>
int MainWindow::sign(const T &x)
{
    if (x >= 0) {
        return 1;
    } else {
        return -1;
    }
}

void MainWindow::on_actionPerception_triggered()
{
    dlgRandomData2 = new DialogRandomData2;

    dlgRandomData2->show();
    connect(dlgRandomData2,
            SIGNAL(sendData(int, vec, mat, int, vec, mat)),
            this,
            SLOT(perception(int, vec, mat, int, vec, mat)));
}

void MainWindow::on_actionMinimum_Distance_Classifier_triggered()
{
    dlgRandomData3 = new DialogRandomData3;

    dlgRandomData3->show();
    connect(dlgRandomData3,
            SIGNAL(sendData(int, vec, mat, int, vec, mat, int, vec, mat)),
            this,
            SLOT(minimumDistanceClassifier(int, vec, mat, int, vec, mat, int, vec, mat)));
}

// 求向量 v 的众数，也就是出现频次最高的那个数
// 输入类型必需为 vec
// 返回值类型为 uword
// 并且假设输入 v 的取值为标签的取值 {0, 1,..., N-1}
// nCount 为类别数量
uword MainWindow::mode(const vec &v,
                       const uword &nCount)
{
    // 显然，只有一个元素，也就是 k = 1 的时候，直接返回就好了
    if (v.n_elem == 1) {
        return v(0);
    } else {
        // linspace(0, max(v), max(v) + 1) 的向量为 (0, 1, 2,..., max(v) - 1)
        // 然后按照这个组距求直方图，histc 返回的是直方图中每个组的频数
        uvec histcV = histc(v, linspace(0, nCount - 1, nCount));
        // 显然，频数最高的那个的下标对应的就是相应的标签了
        // 正好标签的取值为 0, 1, 2,..., N-1，和向量的下标是对应的
        return index_max(histcV);
    }
}

void MainWindow::on_actionKNN_triggered()
{
    dlgRandomData3 = new DialogRandomData3;

    dlgRandomData3->show();
    connect(dlgRandomData3,
            SIGNAL(sendData(int, vec, mat, int, vec, mat, int, vec, mat)),
            this,
            SLOT(KNN(int, vec, mat, int, vec, mat, int, vec, mat)));
}

void MainWindow::fisher(const int &N1,
                        const vec &mu1,
                        const mat &covariance1,
                        const int &N2,
                        const vec &mu2,
                        const mat &covariance2)
{
    // 生成随机数据
    // armadillo 的 mvnrnd 函数与 matlab 的 mvnrnd 函数有区别
    // armadillo 的参数 N 是只生成的矩阵的列数，而 matlab 的指的是行数
    mat data1 = mvnrnd(mu1, covariance1, N1);
    mat data2 = mvnrnd(mu2, covariance2, N2);
    // 拼接到一起，因此这里拼接的时候需要注意转置
    mat data = join_vert(data1.t(), data2.t());
    // 生成对应的标签，注意应该取的是数据的列数
    ivec label1(data1.n_cols, arma::fill::ones);
    label1 = -label1;
    ivec label2(data2.n_cols, arma::fill::ones);
    ivec label = join_vert(label1, label2);

    // 存储输出结果
    vec weight(data.n_cols);
    vec dataProj(label.n_rows);
    double threshold;
    // 训练
    fisherTrain(data, label, weight, dataProj, threshold);
    // 存储测试结果
    ivec predictedLabel(label.n_rows);
    double precision, recall, accuracy, F1;
    // 测试
    fisherTest(data, weight, threshold, label, predictedLabel, precision, recall, accuracy, F1);
    //
    // 结果绘图
    //
    // 类别1，显然这里要用散点图
    QScatterSeries *group1 = new QScatterSeries;
    // 类别2
    QScatterSeries *group2 = new QScatterSeries;

    // 直接遍历，根据 label 取值的不同，将数据点加入到不同的 series
    // 这样的代码会更加通用，样本的数量不再是固定的前50个为类别1，后50个为类别2
    for (uword i = 0; i < label.n_elem; ++i) {
        if (label(i) == -1) {
            group1->append(data(i, 0), data(i, 1));
        } else if (label(i) == 1) {
            group2->append(data(i, 0), data(i, 1));
        } else {
            // something might be wrong, but we don't care
            // and do nothing
        }
    }

    // 设置名称，在图例中显示
    group1->setName(tr("Group 1"));
    group2->setName(tr("Group 2"));
    // 设置 MarkerShape
    group1->setMarkerShape(QScatterSeries::MarkerShapeCircle);
    group2->setMarkerShape(QScatterSeries::MarkerShapeRectangle);
    // 设置颜色
    //    group1->setColor(Qt::GlobalColor::red);
    //    group2->setColor(Qt::GlobalColor::green);

    // 绘图的范围
    // 取特征点的最值在往外增加一个 offset
    double offset = 0.3;
    double xMin = floor(data.col(0).min()) - offset;
    double xMax = ceil(data.col(0).max()) + offset;
    double yMin = floor(data.col(1).min()) - offset;
    double yMax = ceil(data.col(1).max()) + offset;

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
    //chart->setTheme(QChart::ChartThemeBlueIcy);
    chart->setTitle(tr("Fisher Linear Discriminant Analysis"));
    // 启用动画
    chart->setAnimationOptions(QChart::AllAnimations);
    // 这个可以设置动画的时长，默认好像是 1000
    // chart->setAnimationDuration(3000);
    // 图例放在下方，图例中的 MakerShape 直接取自图表
    chart->legend()->setAlignment(Qt::AlignBottom);
    chart->legend()->setMarkerShape(QLegend::MarkerShapeFromSeries);
    chart->setGeometry(ui->graphicsView->rect());

    // 创建一个 QGraphicsScene 对象
    QGraphicsScene *scene = new QGraphicsScene;
    // 将 chart 添加到 scene 中
    scene->addItem(chart);
    // 连接 UI 中的 QGrapicsView 对象与 scene
    ui->graphicsView->setScene(scene);

    // 弹出一个消息框，显示测试结果
    QMessageBox resultBox;
    QString resultString = tr("Precision:\t%1\nRecall:\t\t%2\nAccuracy:\t%3\nF1:\t\t%4").arg(precision).arg(recall).arg(accuracy).arg(F1);
    resultBox.setText(resultString);
    resultBox.setWindowTitle(tr("Fisher Evaluation Result"));
    resultBox.exec();
}

void MainWindow::perception(const int N1,
                            const vec &mu1,
                            const mat &covariance1,
                            const int N2,
                            const vec &mu2,
                            const mat &covariance2)
{
    // 生成随机数据
    mat data1 = mvnrnd(mu1, covariance1, N1);
    mat data2 = mvnrnd(mu2, covariance2, N2);
    // 拼接到一起
    mat data = join_vert(data1.t(), data2.t());
    // 生成对应的标签
    ivec label1(data1.n_cols, arma::fill::ones);
    label1 = -label1;
    ivec label2(data2.n_cols, arma::fill::ones);
    ivec label = join_vert(label1, label2);

    // 权向量
    vec weight(3, arma::fill::randn);
    // 训练
    double learningRate = 0.5;
    uword maxEpoch = 1000;
    perceptionTrain(data, label, learningRate, maxEpoch, weight);

    //
    // 结果绘图，参考 fisher 的结果绘图代码即可
    QScatterSeries *group1 = new QScatterSeries;
    QScatterSeries *group2 = new QScatterSeries;

    for (uword i = 0; i < label.n_elem; ++i) {
        if (label(i) == -1) {
            group1->append(data(i, 0), data(i, 1));
        } else if (label(i) == 1) {
            group2->append(data(i, 0), data(i, 1));
        } else {
            // something might be wrong, but we don't care
            // and do nothing
        }
    }

    group1->setName(tr("Group 1"));
    group2->setName(tr("Group 2"));
    group1->setMarkerShape(QScatterSeries::MarkerShapeCircle);
    group2->setMarkerShape(QScatterSeries::MarkerShapeRectangle);

    double offset = 0.3;
    double xMin = floor(data.col(0).min()) - offset;
    double xMax = ceil(data.col(0).max()) + offset;
    double yMin = floor(data.col(1).min()) - offset;
    double yMax = ceil(data.col(1).max()) + offset;

    QLineSeries *border = new QLineSeries;
    for (double x = xMin; x < 2*xMax; ++x) {
        border->append(x, -weight(0)/weight(1)*x - weight(2)/weight(1));
    }
    border->setName(tr("Class Border"));
    border->setPen(Qt::DashLine);

    QChart *chart = new QChart;
    chart->addSeries(group1);
    chart->addSeries(group2);
    chart->addSeries(border);

    chart->createDefaultAxes();
    chart->axisX()->setRange(xMin, xMax);
    chart->axisY()->setRange(yMin, yMax);
    chart->setTitle(tr("Perception"));
    chart->setAnimationOptions(QChart::AllAnimations);
    chart->legend()->setAlignment(Qt::AlignBottom);
    chart->legend()->setMarkerShape(QLegend::MarkerShapeFromSeries);
    chart->setGeometry(ui->graphicsView->rect());

    QGraphicsScene *scene = new QGraphicsScene;
    scene->addItem(chart);
    ui->graphicsView->setScene(scene);

    // 测试结果
    double precision, recall, accuracy, F1;
    ivec predictedLabel(label.size());

    perceptionTest(data, label, weight, predictedLabel, precision, recall, accuracy, F1);
    // 弹出一个消息框，显示测试结果
    QMessageBox resultBox;
    QString resultString = tr("Precision:\t%1\nRecall:\t\t%2\nAccuracy:\t%3\nF1:\t\t%4").arg(precision).arg(recall).arg(accuracy).arg(F1);
    resultBox.setText(resultString);
    resultBox.setWindowTitle(tr("Perception Evaluation Result"));
    resultBox.exec();
}

void MainWindow::minimumDistanceClassifier(int N1, vec mu1, mat covariance1, int N2, vec mu2, mat covariance2, int N3, vec mu3, mat covariance3)
{
    // 生成随机数据
    mat data1 = mvnrnd(mu1, covariance1, N1);
    mat data2 = mvnrnd(mu2, covariance2, N2);
    mat data3 = mvnrnd(mu3, covariance3, N3);
    // 拼接到一起
    mat data = join_vert(join_vert(data1.t(), data2.t()), data3.t());
    // 标签
    vec label1(data1.n_cols, arma::fill::zeros);
    vec label2(data2.n_cols, arma::fill::ones);
    vec label3(data3.n_cols);
    label3.fill(2);
    vec label = join_vert(join_vert(label1, label2), label3);

    // 将数据和标签拼接在一起，最后一列为标签
    mat dataNlabel = join_horiz(data, label);
    // 打乱顺序
    dataNlabel = shuffle(dataNlabel);
    // 训练样本数，简单一点，只用最后一个样本作为测试
    const uword trainNum = dataNlabel.n_rows - 1;
    // 类别数
    const int nCount = 3;

    // 训练与测试数据及其标签
    mat trainData = dataNlabel(span(0, trainNum - 1), span(0, dataNlabel.n_cols - 2));
    Row<double> testData = dataNlabel(dataNlabel.n_rows - 1, span(0, dataNlabel.n_cols - 2));
    // 因为后面需要做标签的比较，所以还是得用整型，用浮点型无法比较 ==
    vec trainLabelDouble = dataNlabel(span(0, trainNum - 1), dataNlabel.n_cols - 1);
    uvec trainLabel(trainLabelDouble.n_rows, trainLabelDouble.n_cols);
    for (uword i = 0; i < trainNum; ++i) {
        trainLabel(i) = static_cast<uword>(trainLabelDouble(i));
    }
    // uword testLabel = dataNlabel(dataNlabel.n_rows - 1, dataNlabel.n_cols - 1);

    // 计算均值
    // 一共是 nCount 个类别，data.n_cols 个特征
    // 所以用 nCount x data.n_cols 的矩阵来保存
    mat meanValue(nCount, dataNlabel.n_cols - 1);
    // 找到 label == i 的训练样本，并计算他们的均值
    // 也就是计算每一类样本的距离中心
    for (uword i = 0; i < nCount; ++i) {
        meanValue.row(i) = mean(trainData.rows(arma::find(trainLabel == i)));
    }

    // 计算测试样本到各类别的距离，并存储到 distance 向量中
    // 距离最小的那个的下标对应的就是预测的标签
    vec distance(nCount);
    for (uword i = 0; i < nCount; ++i) {
        distance(i) = norm(testData - meanValue.row(i));
    }
    uword predictedLabel = distance.index_min();

    // 结果绘图
    QScatterSeries *group1 = new QScatterSeries;
    QScatterSeries *group2 = new QScatterSeries;
    QScatterSeries *group3 = new QScatterSeries;
    QScatterSeries *groupTest = new QScatterSeries;
    QScatterSeries *groupMean1 = new QScatterSeries;
    QScatterSeries *groupMean2 = new QScatterSeries;
    QScatterSeries *groupMean3 = new QScatterSeries;
    QLineSeries *line1 = new QLineSeries;
    QLineSeries *line2 = new QLineSeries;
    QLineSeries *line3 = new QLineSeries;

    // 读取数据，保存到对应的 series 里
    // 训练数据
    for (uword i = 0; i < trainNum; ++i) {
        if (trainLabel(i) == 0) {
            group1->append(trainData(i, 0), trainData(i, 1));
        } else if (trainLabel(i) == 1) {
            group2->append(trainData(i, 0), trainData(i, 1));
        } else if (trainLabel(i) == 2) {
            group3->append(trainData(i, 0), trainData(i, 1));
        } else {
            // something might be wrong, but we don't care
            // and do nothing
        }
    }
    // 训练样本
    groupTest->append(testData(0, 0), testData(0, 1));
    group1->setColor(Qt::GlobalColor::red);
    group2->setColor(Qt::GlobalColor::green);
    group3->setColor(Qt::GlobalColor::blue);
    // 根据预测标签决定测试样本的颜色
    if (predictedLabel == 0) {
        groupTest->setColor(group1->color());
    } else if (predictedLabel == 1) {
        groupTest->setColor(group2->color());
    } else if (predictedLabel == 2) {
        groupTest->setColor(group3->color());
    } else {
    // do nothing
    }
    // 每个类别的中心点
    groupMean1->append(meanValue(0, 0), meanValue(0, 1));
    groupMean2->append(meanValue(1, 0), meanValue(1, 1));
    groupMean3->append(meanValue(2, 0), meanValue(2, 1));
    // 中心点到测试样本的连接线
    line1->append(meanValue(0, 0), meanValue(0, 1));
    line1->append(testData(0, 0), testData(0, 1));
    line2->append(meanValue(1, 0), meanValue(1, 1));
    line2->append(testData(0, 0), testData(0, 1));
    line3->append(meanValue(2, 0), meanValue(2, 1));
    line3->append(testData(0, 0), testData(0, 1));

    // 换个 markerShape 和大小，这样才好对比
    const double factor = 1.5f;
    groupTest->setMarkerShape(QScatterSeries::MarkerShapeRectangle);
    groupTest->setMarkerSize(factor*group1->markerSize());
    groupMean1->setMarkerSize(factor*group1->markerSize());
    groupMean2->setMarkerSize(factor*group2->markerSize());
    groupMean3->setMarkerSize(factor*group3->markerSize());
    // 设置类别中心点的颜色
    groupMean1->setColor(group1->color());
    groupMean2->setColor(group2->color());
    groupMean3->setColor(group3->color());
    // 连线的颜色用与类别相同的颜色
    line1->setColor(group1->color());
    line2->setColor(group2->color());
    line3->setColor(group3->color());

    // 设置名称
    group1->setName(tr("Group1"));
    group2->setName(tr("Group2"));
    group3->setName(tr("Group3"));
    groupTest->setName(tr("Test"));

    QChart *chart = new QChart;
    chart->addSeries(group1);
    chart->addSeries(group2);
    chart->addSeries(group3);
    chart->addSeries(groupMean1);
    chart->addSeries(groupMean2);
    chart->addSeries(groupMean3);
    chart->addSeries(groupTest);
    chart->addSeries(line1);
    chart->addSeries(line2);
    chart->addSeries(line3);
    // 隐藏不需要的图例
    chart->legend()->markers(groupMean1)[0]->setVisible(false);
    chart->legend()->markers(groupMean2)[0]->setVisible(false);
    chart->legend()->markers(groupMean3)[0]->setVisible(false);
    chart->legend()->markers(line1)[0]->setVisible(false);
    chart->legend()->markers(line2)[0]->setVisible(false);
    chart->legend()->markers(line3)[0]->setVisible(false);

    // 绘图的范围
    // 取特征点的最值在往外增加一个 offset
    double offset = 0.3;
    double xMin = floor(data.col(0).min()) - offset;
    double xMax = ceil(data.col(0).max()) + offset;
    double yMin = floor(data.col(1).min()) - offset;
    double yMax = ceil(data.col(1).max()) + offset;

    // 创建默认的坐标轴
    chart->createDefaultAxes();
    // 设置坐标轴的范围
    chart->axisX()->setRange(xMin, xMax);
    chart->axisY()->setRange(yMin, yMax);

    // 不能设置主题，否则会覆盖之前的颜色设定
    // chart->setTheme(QChart::ChartThemeBlueIcy);
    chart->setTitle(tr("Minimum Distance Classifier"));
    // 启用动画
    chart->setAnimationOptions(QChart::AllAnimations);
    // 图例放在下方，图例中的 MakerShape 直接取自图表
    chart->legend()->setAlignment(Qt::AlignBottom);
    chart->legend()->setMarkerShape(QLegend::MarkerShapeFromSeries);
    // 图表大小为 graphicsView 的大小
    chart->setGeometry(ui->graphicsView->rect());

    // 创建一个 QGraphicsScene 对象
    QGraphicsScene *scene = new QGraphicsScene;
    // 将 chart 添加到 scene 中
    scene->addItem(chart);
    // 连接 UI 中的 QGrapicsView 对象与 scene
    ui->graphicsView->setScene(scene);
}

void MainWindow::KNN(const int &N1, const vec &mu1, const mat &covariance1, const int &N2, const vec &mu2, const mat &covariance2, const int &N3, const vec &mu3, const mat &covariance3)
{
    // 生成随机数据
    mat data1 = mvnrnd(mu1, covariance1, N1);
    mat data2 = mvnrnd(mu2, covariance2, N2);
    mat data3 = mvnrnd(mu3, covariance3, N3);
    // 拼接到一起
    mat data = join_vert(join_vert(data1.t(), data2.t()), data3.t());
    // 标签
    vec label1(data1.n_cols, arma::fill::zeros);
    vec label2(data2.n_cols, arma::fill::ones);
    vec label3(data3.n_cols);
    label3.fill(2);
    vec label = join_vert(join_vert(label1, label2), label3);

    // 将数据和标签拼接在一起，最后一列为标签
    mat dataNlabel = join_horiz(data, label);
    // 打乱顺序
    dataNlabel = shuffle(dataNlabel);
    // 训练样本数，简单一点，只用最后一个样本作为测试
    const uword trainNum = dataNlabel.n_rows - 1;
    // 类别数
    const int nCount = 3;
    // 训练样本
    mat trainData = dataNlabel.rows(0, trainNum - 1);
    // 测试样本
    Row<double> testData = dataNlabel.row(dataNlabel.n_rows - 1);

    // 测试样本到训练样本的欧式距离
    vec euclideanDistance(trainNum);
    for (uword i = 0; i < trainNum; ++i) {
        euclideanDistance(i) = norm(trainData(i, span(0, 1)) - testData.cols(0, 1));
    }
    // 将距离和数据放在一起，写成增广矩阵
    mat distance = join_horiz(euclideanDistance, trainData);
    // 升序排序
    // 将矩阵按照第一列排序，这样对应的标签也会跟着排序
    distance = sortRows(distance);
    // 从用户输入读取邻居个数 k，默认值为 3，最小值为 1，最大值为 10
    const int kMax = std::max(std::max(N1, N2), N3);
    const int kMin = 1;
    const int step = 1;
    const int kDefault = 3;
    int k = QInputDialog::getInt(this,
                                 tr("K Neighboor Setting"),
                                 tr("K Neighboor:"),
                                 kDefault,
                                 kMin,
                                 kMax,
                                 step);

    // 排序后将 k 个邻居保存到变量 kNeigborData 中
    mat kNeighborData = distance(span(0, k - 1), span(1, distance.n_cols - 1));
    // 找到前 k 个值里对应标签出现次数最多的就是预测的标签
    uword predictedLabel = mode(kNeighborData.col(kNeighborData.n_cols - 1), nCount);

    // 结果绘图
    QScatterSeries *group1 = new QScatterSeries;
    QScatterSeries *group2 = new QScatterSeries;
    QScatterSeries *group3 = new QScatterSeries;
    QScatterSeries *groupTest = new QScatterSeries;
    QScatterSeries *groupKNeighbor = new QScatterSeries;
    // 读取数据，保存到对应的 series 里
    // 训练数据
    for (uword i = 0; i < trainNum; ++i) {
        if (static_cast<uword>(trainData(i, trainData.n_cols - 1)) == 0) {
            group1->append(trainData(i, 0), trainData(i, 1));
        } else if (static_cast<uword>(trainData(i, trainData.n_cols - 1)) == 1) {
            group2->append(trainData(i, 0), trainData(i, 1));
        } else if (static_cast<uword>(trainData(i, trainData.n_cols - 1)) == 2) {
            group3->append(trainData(i, 0), trainData(i, 1));
        } else {
            // something might be wrong, but we don't care
            // and do nothing
        }
    }
    // 训练样本和测试样本
    groupTest->append(testData(0, 0), testData(0, 1));
    group1->setColor(Qt::GlobalColor::red);
    group2->setColor(Qt::GlobalColor::green);
    group3->setColor(Qt::GlobalColor::blue);
    // 根据预测标签决定测试样本的颜色
    if (predictedLabel == 0) {
        groupTest->setColor(group1->color());
    } else if (predictedLabel == 1) {
        groupTest->setColor(group2->color());
    } else if (predictedLabel == 2) {
        groupTest->setColor(group3->color());
    } else {
    // do nothing
    }
    groupTest->setMarkerShape(QScatterSeries::MarkerShapeRectangle);
    const qreal factor = 1.5f;
    groupTest->setMarkerSize(factor*groupTest->markerSize());
    // K 个邻居
    for (uword i = 0; i < kNeighborData.n_rows; ++i) {
        groupKNeighbor->append(kNeighborData(i, 0), kNeighborData(i, 1));
    }
    // 因为邻居点与训练样本点是重合的，所以设置一下透明色
    groupKNeighbor->setOpacity(0.5);
    groupKNeighbor->setBorderColor(Qt::GlobalColor::cyan);
    groupKNeighbor->setMarkerShape(QScatterSeries::MarkerShapeRectangle);
    groupKNeighbor->setMarkerSize(factor*group1->markerSize());
    // 设置名称
    group1->setName(tr("Group1"));
    group2->setName(tr("Group2"));
    group3->setName(tr("Group3"));
    groupTest->setName(tr("Test"));
    groupKNeighbor->setName(tr("Neighbor"));

    QChart *chart = new QChart;
    chart->addSeries(group1);
    chart->addSeries(group2);
    chart->addSeries(group3);
    chart->addSeries(groupTest);
    chart->addSeries(groupKNeighbor);

    // 绘图的范围
    // 取特征点的最值在往外增加一个 offset
    double offset = 0.3;
    double xMin = floor(trainData.col(0).min()) - offset;
    double xMax = ceil(trainData.col(0).max()) + offset;
    double yMin = floor(trainData.col(1).min()) - offset;
    double yMax = ceil(trainData.col(1).max()) + offset;

    // 创建默认的坐标轴
    chart->createDefaultAxes();
    // 设置坐标轴的范围
    chart->axisX()->setRange(xMin, xMax);
    chart->axisY()->setRange(yMin, yMax);
    // 不能设置主题了，否则会覆盖之前的颜色定义
    // chart->setTheme(QChart::ChartThemeBlueIcy);
    chart->setTitle(tr("KNN"));
    // 启用动画
    chart->setAnimationOptions(QChart::AllAnimations);
    // 这个可以设置动画的时长，默认好像是 1000
    // chart->setAnimationDuration(3000);
    // 图例放在下方，图例中的 MakerShape 直接取自图表
    chart->legend()->setAlignment(Qt::AlignBottom);
    chart->legend()->setMarkerShape(QLegend::MarkerShapeFromSeries);
    chart->setGeometry(ui->graphicsView->rect());

    // 创建一个 QGraphicsScene 对象
    QGraphicsScene *scene = new QGraphicsScene;
    // 将 chart 添加到 scene 中
    scene->addItem(chart);
    // 连接 UI 中的 QGrapicsView 对象与 scene
    ui->graphicsView->setScene(scene);
}

// K 均值聚类，实现方法参考 http://aqibsaeed.github.io/2016-06-24-k-means/
// 主要步骤：
// 1. 随机选择初始聚类中心
// 2. 根据样本点到聚类中心的距离
// 将样本点分给不同的聚类
// 3. 重新计算聚类中心（用均值作为聚类中心）
// 4. 重复步骤2和3，直到收敛
// 收敛条件是聚类中心不再变化或者变化很小
// 也是设置最大的迭代次数
void MainWindow::kMeans(const int &N1,
                        const vec &mu1,
                        const mat &covariance1,
                        const int &N2,
                        const vec &mu2,
                        const mat &covariance2,
                        const int &N3,
                        const vec &mu3,
                        const mat &covariance3)
{
    // 生成随机数据
    mat data1 = mvnrnd(mu1, covariance1, N1);
    mat data2 = mvnrnd(mu2, covariance2, N2);
    mat data3 = mvnrnd(mu3, covariance3, N3);
    // 拼接到一起
    mat data = join_vert(join_vert(data1.t(), data2.t()), data3.t());
    // 标签
    vec label1(data1.n_cols, arma::fill::zeros);
    vec label2(data2.n_cols, arma::fill::ones);
    vec label3(data3.n_cols);
    label3.fill(2);
    vec label = join_vert(join_vert(label1, label2), label3);
    // 将数据和标签拼接在一起，最后一列为标签
    mat dataNlabel = join_horiz(data, label);

    // 将原始数据绘制出来
    QScatterSeries *groupGroundTruth[3];
    // 创建 3 个 group
    for (uword i = 0; i < 3; ++i) {
        groupGroundTruth[i] = new QScatterSeries;
        groupGroundTruth[i]->setName(tr("Group%1").arg(i));
    }
    // 根据 label 添加到不同的 group 中
    for (uword i = 0; i < dataNlabel.n_rows; ++i) {
        for (uword j = 0; j < 3; ++j) {
            if (static_cast<uword>(dataNlabel(i, dataNlabel.n_cols - 1) == j)) {
                groupGroundTruth[j]->append(dataNlabel(i, 0), dataNlabel(i, 1));
            }
        }
    }
    QChart *chart = new QChart;
    for (uword i = 0; i < 3; ++i) {
        chart->addSeries(groupGroundTruth[i]);
    }
    chart->setTheme(QChart::ChartThemeBlueIcy);
    // 绘图的范围
    // 取特征点的最值在往外增加一个 offset
    double offset = 0.3;
    double xMin = floor(dataNlabel.col(0).min()) - offset;
    double xMax = ceil(dataNlabel.col(0).max()) + offset;
    double yMin = floor(dataNlabel.col(1).min()) - offset;
    double yMax = ceil(dataNlabel.col(1).max()) + offset;

    chart->createDefaultAxes();
    chart->axisX()->setRange(xMin, xMax);
    chart->axisY()->setRange(yMin, yMax);
    chart->setTitle(tr("K Means Cluster - Original Data"));
    chart->setAnimationOptions(QChart::AllAnimations);
    chart->legend()->setAlignment(Qt::AlignBottom);
    chart->legend()->setMarkerShape(QLegend::MarkerShapeFromSeries);
    chart->setGeometry(ui->graphicsView->rect());

    QGraphicsScene *scene = new QGraphicsScene;
    scene->addItem(chart);
    ui->graphicsView->setScene(scene);

    // 从用户处读取聚类的类别数，默认为 3
    const int kMax = 10;
    const int kMin = 2;
    const int step = 1;
    const int kDefault = 3;
    uword k = QInputDialog::getInt(this,
                                 tr("K Means Cluster Setting"),
                                 tr("Cluster number:"),
                                 kDefault,
                                 kMin,
                                 kMax,
                                 step);

    // 打乱顺序
    dataNlabel = shuffle(dataNlabel);
    // 设置初始的聚类中心，直接取前 k 个样本
    mat centroids = dataNlabel(span(0, k - 1), span(0, 1));
    //
    const uword maxEpoch = 1024;
    for (uword epoch = 0; epoch < maxEpoch; ++epoch) {
        // 计算各个点到聚类中心的距离
        // 用一个 dataNlabel.n_rows x k 的矩阵保存所有样本点到聚类中心的距离
        mat euclideanDistance(dataNlabel.n_rows, k);
        for (uword i = 0; i < dataNlabel.n_rows; ++i) {
            for (uword j = 0; j < k; ++j) {
                euclideanDistance(i, j) = norm(dataNlabel(i, span(0, 1)) - centroids(j, span(0, 1)));
            }
        }
        // 根据各个样本点到聚类中心的距离分类
        // 也就是更新 label
        // 求最小值对应的下标
        uvec indMin = index_min(euclideanDistance, 1);
        // 根据下标更新 label
        for (uword i = 0; i < dataNlabel.n_rows; ++i) {
            dataNlabel(i, dataNlabel.n_cols - 1) = static_cast<double>(indMin(i));
        }
        // 重新计算聚类中心
        // 标签取值为 0, 1, 2，所以只需要找到 label < 0.5, label > 0.5 && label < 1.5 和 label > 1.5
        // 的样本就对了．不能直接使用 == 作为判断条件是因为这是浮点数
        for (uword i = 0; i < k; ++i) {
            const double offset = 0.5;
            uvec tmpIndRow = arma::find(dataNlabel.col(dataNlabel.n_cols - 1) > i - offset &&
                                     dataNlabel.col(dataNlabel.n_cols - 1) < i + offset);
            uvec tmpIndCol = {0, 1};
            mat tmpData = dataNlabel(span(0, dataNlabel.n_rows - 1), span(0, 1));
            centroids.row(i) = mean(tmpData.submat(tmpIndRow, tmpIndCol));
        }
        // 最后再把数据打乱
        dataNlabel = shuffle(dataNlabel);
    }
    // 结果绘图
    QScatterSeries *group[k];
    // 创建 k 个 group
    for (uword i = 0; i < k; ++i) {
        group[i] = new QScatterSeries;
    }
    for (uword i = 0; i < dataNlabel.n_rows; ++i) {
        for (uword j = 0; j < k; ++j) {
            if (static_cast<uword>(dataNlabel(i, dataNlabel.n_cols - 1) == j)) {
                group[j]->append(dataNlabel(i, 0), dataNlabel(i, 1));
            }
        }
    }

    // 聚类中心
    QScatterSeries *groupCentroids = new QScatterSeries;
    for (uword i = 0; i < centroids.n_rows; ++i) {
        groupCentroids->append(centroids(i, 0), centroids(i, 1));
    }

    for (uword i = 0; i < k; ++i) {
        group[i]->setName(tr("Group%1").arg(i+1));
    }
    groupCentroids->setName(tr("Centroids"));
    groupCentroids->setMarkerShape(QScatterSeries::MarkerShapeRectangle);
    const qreal factor = 1.5f;
    groupCentroids->setMarkerSize(factor*groupCentroids->markerSize());

    QChart *chartResult = new QChart;
    for (uword i = 0; i < k; ++i) {
        chartResult->addSeries(group[i]);
    }
    chartResult->addSeries(groupCentroids);
    chartResult->setTheme(QChart::ChartThemeBlueIcy);

    // 创建默认的坐标轴
    chartResult->createDefaultAxes();
    // 设置坐标轴的范围
    chartResult->axisX()->setRange(xMin, xMax);
    chartResult->axisY()->setRange(yMin, yMax);
    chartResult->setTitle(tr("K Means Cluster"));
    // 启用动画
    chartResult->setAnimationOptions(QChart::AllAnimations);
    // 图例放在下方，图例中的 MakerShape 直接取自图表
    chartResult->legend()->setAlignment(Qt::AlignBottom);
    chartResult->legend()->setMarkerShape(QLegend::MarkerShapeFromSeries);
    chartResult->setGeometry(ui->graphicsView->rect());

    // 删除之前的 chart
    scene->removeItem(chart);
    // 将 chart 添加到 scene 中
    scene->addItem(chartResult);
    // 连接 UI 中的 QGrapicsView 对象与 scene
    ui->graphicsView->setScene(scene);
}

// 将矩阵 data 按照第一列排序，其他列的行跟第一列保持一致
template <typename T>
Mat<T> MainWindow::sortRows(const Mat<T> &data)
{
    Mat<T> result(arma::size(data));

    // 第一列直接排序
    result.col(0) = sort(data.col(0));
    // 取得排序的下标
    uvec ind = stable_sort_index(data.col(0));
    // 临时变量，因为好像直接用 data.col(i)(ind) 会有语法错误？
    Col<T> tmp;
    // 遍历第 1 到最后一列，根据第一列的排序下标排序
    for (uword i = 1; i < data.n_cols; ++i) {
        tmp = data.col(i);
        result.col(i) = tmp(ind);
    }

    return result;
}

void MainWindow::on_actionK_Means_triggered()
{
    dlgRandomData3 = new DialogRandomData3;

    dlgRandomData3->show();
    connect(dlgRandomData3,
            SIGNAL(sendData(int, vec, mat, int, vec, mat, int, vec, mat)),
            this,
            SLOT(kMeans(int, vec, mat, int, vec, mat, int, vec, mat)));
}
