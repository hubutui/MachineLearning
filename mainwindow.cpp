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
int MainWindow::rgbToGray(const int &r, const int &g, const int &b)
{
    return (r * 11 + g * 16 + b * 5)/32;
}

void MainWindow::fisherTrain(const mat &data, const ivec &label, vec &weight, vec &dataProj, double &threshold)
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

void MainWindow::fisherTest(const mat &data, const vec &weight, const double &threshold, const ivec &label,
                            ivec &predictedLabel, double &precision, double &recall, double &accuracy, double &F1)
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
void MainWindow::perceptionTrain(const mat &data, const ivec &label, const double &learningRate, const int &maxEpoch, vec &weight)
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
void MainWindow::perceptionTest(const mat &data, const ivec &label, const vec &weight,
                                ivec &predictedLabel, double &precision, double &recall, double &accuracy, double &F1)
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

// 最小距离分类器
// data 和 label 分别为输入数据和标签
// trainRate 表示将多少数据当作已知的，剩余的数据用于分类
// nCount 为类别数
// predictedLabel 是预测的 label
// medianPoint 是每个类的中心点，打算用来绘制分类界线
// 分类界线为两个类的中心点连线的垂直平分线
void MainWindow::minDistanceClassifier(const mat &data,
                                       const uvec &label,
                                       const double &trainRate,
                                       const unsigned int nCount,
                                       uvec &predictedLabel,
                                       double &accuracy)
{
    const int trainNum = data.n_rows*trainRate;
    const int testNum = data.n_rows - trainNum;

    mat trainData = data.rows(0, trainNum - 1);
    mat testData = data.rows(trainNum, data.n_rows - 1);
    uvec trainLabel = label.rows(0, trainNum - 1);
    uvec testLabel = label.rows(trainNum, data.n_rows - 1);

    // 计算均值
    // 一共是 nCount 个类别，data.n_cols 个特征
    // 所以用 nCount x data.n_cols 的矩阵来保存
    mat meanValue(nCount, data.n_cols);

    // 找到 label == i 的训练数据，并计算它们的均值
    uword siz = trainData.n_cols;
    siz = trainData.n_rows;
    siz = meanValue.n_cols;
    siz = meanValue.n_rows;
    for (uword i = 0; i < nCount; ++i) {
        meanValue.row(i) = mean(trainData.rows(arma::find(trainLabel == i)));
    }

    // 计算测试样本与各类别的距离
    vec distance(nCount);
    for (uword i = 0; i < testData.n_rows; ++i) {
        for (unsigned int j = 0; j < nCount; ++j) {
            distance(j) = norm(testData.row(i) - meanValue.row(j));
        }
        // 最小距离的下标就是其预测所属的类别
        predictedLabel(i) = distance.index_min();
    }

    // 计算 predictedLabel 与 testLabel 相同的个数就是分类正确的个数
    // 然后就可以计算正确率

    double correctNum = sum(testLabel == predictedLabel);
    accuracy = correctNum/testNum;
}

// 读取 csv 文件
// 将结果按行优先顺序存储到 data 中
// 这个本来是用来从 csv 文件读取数据的，后来发现可以用 armadillo 读取 matlab 保存的文件
// 然后就用不上了．
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
    ui->graphicsView_in->setScene(scene);

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
    // 使用 iris 数据集，从纯文本文件读入
    // 文件的格式为 tab 分隔的数值
    // 按列算，前面的列为特征，最后一列为标签
    // 这里都是只用两个特征，所以只读取前两列和最后一列
    // 标签的取值为 {-1, 1}
    // 这种类型的文件可以直接用 MATLAB/Octave 的 save 函数保存
    // 例如 save('iris.txt', 'data', '-ascii')
    QString dataFile = QFileDialog::getOpenFileName(
                this, tr("Open data file"), QDir::currentPath());
    if (dataFile.isEmpty()) {
        QMessageBox::critical(this, tr("Error!"), tr("Please select a valid data file."));
        return;
    }
    mat data;
    data.load(dataFile.toStdString().data());

    mat features = data.cols(0, 1);
    // 读取最后一列做为标签
    vec tmp = data.col(data.n_cols - 1);
    ivec label(tmp.size());

    for (uword i = 0; i < tmp.size(); ++i) {
        label(i) = tmp(i);
    }
    // 读取数据完毕

    //    // 生成随机数据
    // 实际上这里的代码应该用不上了，因为显然可以用 matlab 生成随机数据
    // 然后保存到文件，再用上面的代码读取就好了
    //    mat features(100, 2, arma::fill::randn);
    //    uvec label(100, arma::fill::ones);
    //    for (uword i = 0; i < label.size()/2; ++i) {
    //        label(i) = 0;
    //    }
    //    // 随机数据生成完毕

    // 存储输出结果
    vec weight(features.n_cols);
    vec dataProj(label.size());
    double threshold;

    fisherTrain(features, label, weight, dataProj, threshold);

    // 存储测试结果
    ivec predictedLabel(label.size());
    double precision, recall, accuracy, F1;

    fisherTest(features, weight, threshold, label, predictedLabel, precision, recall, accuracy, F1);

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
            group1->append(features(i, 0), features(i, 1));
        } else if (label(i) == 1) {
            group2->append(features(i, 0), features(i, 1));
        } else {
            // something might be wrong, but we don't care
            // and do nothing
        }
    }

    // 设置名称，在图例中显示
    group1->setName(tr("Iris Setosa"));
    // 设置 marker 为 10，默认为 15
    group1->setMarkerSize(10);
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
    chart->setGeometry(ui->graphicsView_in->rect());

    // 创建一个 QGraphicsScene 对象
    QGraphicsScene *scene = new QGraphicsScene;
    // 将 chart 添加到 scene 中
    scene->addItem(chart);
    // 连接 UI 中的 QGrapicsView 对象与 scene
    ui->graphicsView_in->setScene(scene);

    // 弹出一个消息框，显示测试结果
    QMessageBox resultBox;
    QString resultString = tr("Precision:\t%1\nRecall:\t%2\nAccuracy:\t%3\nF1:\t%4").arg(precision).arg(recall).arg(accuracy).arg(F1);
    resultBox.setText(resultString);
    resultBox.setWindowTitle(tr("Fisher LDA"));
    resultBox.exec();
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

// 符号函数
// x >=0 时返回 1，否则返回 -1
template<typename T>
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
    // 使用 iris 数据集，从纯文本文件读入
    // 文件的格式为 tab 分隔的数值
    // 按列算，前面的列为特征，最后一列为标签
    // 这里都是只用两个特征，所以只读取前两列和最后一列
    // 标签的取值为 {-1, 1}
    // 这种类型的文件可以直接用 MATLAB/Octave 的 save 函数保存
    // 例如 save('iris.txt', 'data', '-ascii')
    QString dataFile = QFileDialog::getOpenFileName(
                this, tr("Open data file"), QDir::currentPath());
    if (dataFile.isEmpty()) {
        QMessageBox::critical(this, tr("Error!"), tr("Please select a valid data file."));
        return;
    }
    mat data;
    data.load(dataFile.toStdString().data());

    mat features = data.cols(0, 1);
    vec tmp = data.col(data.n_cols - 1);
    ivec label(tmp.size());

    for (uword i = 0; i < tmp.size(); ++i) {
        label(i) = tmp(i);
    }

    // 权向量
    vec weight(3, arma::fill::randn);
    // 训练
    double learningRate = 0.5;
    int maxEpoch = 1000;
    perceptionTrain(features, label, learningRate, maxEpoch, weight);

    // 结果绘图，这部分的代码直接从 fisher 线性判别的那部分修改过来的
    //
    // 类别1，显然这里要用散点图
    QScatterSeries *group1 = new QScatterSeries;
    // 类别2
    QScatterSeries *group2 = new QScatterSeries;

    // 直接遍历，根据 label 取值的不同，将数据点加入到不同的 series
    // 这样的代码会更加通用，样本的数量不再是固定的前50个为类别1，后50个为类别2
    for (uword i = 0; i < label.n_elem; ++i) {
        if (label(i) == -1) {
            group1->append(features(i, 0), features(i, 1));
        } else if (label(i) == 1) {
            group2->append(features(i, 0), features(i, 1));
        }  else {
            // something might be wrong, but we don't care
            // and do nothing
        }
    }

    // 设置名称，在图例中显示
    group1->setName(tr("Iris Setosa"));
    // 设置 marker 为 10，默认为 15
    group1->setMarkerSize(10);
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
        border->append(x, -weight(0)/weight(1)*x - weight(2)/weight(1));
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
    chart->setTitle(tr("Perception"));
    // 启用动画
    chart->setAnimationOptions(QChart::AllAnimations);
    // 这个可以设置动画的时长，默认好像是 1000
    // chart->setAnimationDuration(3000);
    // 图例放在下方，图例中的 MakerShape 直接取自图表
    chart->legend()->setAlignment(Qt::AlignBottom);
    chart->legend()->setMarkerShape(QLegend::MarkerShapeFromSeries);
    chart->setGeometry(ui->graphicsView_in->rect());

    // 创建一个 QGraphicsScene 对象
    QGraphicsScene *scene = new QGraphicsScene;
    // 将 chart 添加到 scene 中
    scene->addItem(chart);
    // 连接 UI 中的 QGrapicsView 对象与 scene
    ui->graphicsView_in->setScene(scene);

    double precision, recall, accuracy, F1;
    ivec predictedLabel(label.size());

    perceptionTest(features, label, weight, predictedLabel, precision, recall, accuracy, F1);
    // 弹出一个消息框，显示测试结果
    QMessageBox resultBox;
    QString resultString = tr("Precision:\t%1\nRecall:\t%2\nAccuracy:\t%3\nF1:\t%4").arg(precision).arg(recall).arg(accuracy).arg(F1);
    resultBox.setText(resultString);
    // resultBox.setWindowTitle(tr("Perception"));
    resultBox.exec();
}

void MainWindow::on_actionMinimum_Distance_Classifier_triggered()
{
    // 读取数据
    QString dataFile = QFileDialog::getOpenFileName(
                this, tr("Open data file"), QDir::currentPath());
    if (dataFile.isEmpty()) {
        QMessageBox::critical(this, tr("Error!"), tr("Please select a valid data file."));
        return;
    }
    mat data;
    data.load(dataFile.toStdString().data());

    // 二维数据，所以就只使用前两个特征
     mat features = data.cols(0, 1);
    // 试试使用所有的特征
//    mat features = data.cols(0, data.n_cols - 2);
    // 训练比率
    const double trainRate = 0.7;
    int trainNum = trainRate*data.n_rows;
    vec tmp = data.col(data.n_cols - 1);
    uvec label(tmp.size());

    for (uword i = 0; i < tmp.size(); ++i) {
        label(i) = tmp(i);
    }

    // 标签取值为 {0, 1,..., N-1}
    // 类别数量为 N
    const unsigned int nCount = label.max() +1;
    // 预测标签，就是测试的数据
    uvec predictedLabel(label.size() - trainNum);
    // 测试结果
    double accuracy;
    // 仅作二分类和三分类
    if (nCount != 2 && nCount != 3) {
        QMessageBox::critical(this, tr("Error"), tr("Only 2 or 3 groups."));
        return;
    }
    // 调用函数，进行分类
    minDistanceClassifier(features, label, trainRate, nCount, predictedLabel, accuracy);

    // 结果绘图
    QScatterSeries *group1 = new QScatterSeries;
    QScatterSeries *group2 = new QScatterSeries;

    QScatterSeries *group1Test = new QScatterSeries;
    QScatterSeries *group2Test = new QScatterSeries;
    QScatterSeries *group3 = new QScatterSeries;
    QScatterSeries *group3Test = new QScatterSeries;

    // 读取数据，保存到对应的 series 里
    // 训练数据
    for (uword i = 0; i < trainNum; ++i) {
        if (label(i) == 0) {
            group1->append(features(i, 0), features(i, 1));
        } else if (label(i) == 1) {
            group2->append(features(i, 0), features(i, 1));
        } else if (label(i) == 2) {
            group3->append(features(i, 0), features(i, 1));
        }
        else {
            // something might be wrong, but we don't care
            // and do nothing
        }
    }

    // 测试数据
    for (uword i = trainNum; i < label.n_elem; ++i) {
        if (label(i) == 0) {
            group1Test->append(features(i, 0), features(i, 1));
        } else if (label(i) == 1) {
            group2Test->append(features(i, 0), features(i, 1));
        } else if (label(i) == 2) {
            group3Test->append(features(i, 0), features(i, 1));
        }
        else {
            // do nothing
        }
    }

    // 设置名称
    group1->setName(tr("Group1"));
    group2->setName(tr("Group2"));
    group3->setName(tr("Group3"));
    group1Test->setName(tr("Group1 Test"));
    group2Test->setName(tr("Group2 Test"));
    group3Test->setName(tr("Group3 Test"));
    // 设置 Marker
    group1->setMarkerSize(10);
    group2->setMarkerSize(10);
    group3->setMarkerSize(10);
    group1Test->setMarkerSize(15);
    group2Test->setMarkerSize(15);
    group3Test->setMarkerSize(15);

    group1->setColor(Qt::GlobalColor::red);
    group2->setColor(Qt::GlobalColor::green);
    group3->setColor(Qt::GlobalColor::blue);
    group1Test->setColor(group1->color());
    group2Test->setColor(group2->color());
    group3Test->setColor(group3->color());

    QChart *chart = new QChart;
    chart->addSeries(group1);
    chart->addSeries(group2);
    chart->addSeries(group1Test);
    chart->addSeries(group2Test);

    if (nCount == 3) {
        chart->addSeries(group3);
        chart->addSeries(group3Test);
    }

    // 绘图的范围
    // 取特征点的最值在往外增加一个 offset
    double offset = 0.3;
    double xMin = floor(features.col(0).min()) - offset;
    double xMax = ceil(features.col(0).max()) + offset;
    double yMin = floor(features.col(1).min()) - offset;
    double yMax = ceil(features.col(1).max()) + offset;

    // 创建默认的坐标轴
    chart->createDefaultAxes();
    // 设置坐标轴的范围
    chart->axisX()->setRange(xMin, xMax);
    chart->axisY()->setRange(yMin, yMax);
    // 不能设置主题了，否则会覆盖之前的颜色定义
    // chart->setTheme(QChart::ChartThemeBlueIcy);
    chart->setTitle(tr("Perception"));
    // 启用动画
    chart->setAnimationOptions(QChart::AllAnimations);
    // 这个可以设置动画的时长，默认好像是 1000
    // chart->setAnimationDuration(3000);
    // 图例放在下方，图例中的 MakerShape 直接取自图表
    chart->legend()->setAlignment(Qt::AlignBottom);
    chart->legend()->setMarkerShape(QLegend::MarkerShapeFromSeries);

    ui->chartView->setChart(chart);
}
