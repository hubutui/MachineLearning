#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "dialogglcm.h"
#include <QMainWindow>
#include <QGraphicsScene>
#include <QPixmap>
#include <QGraphicsPixmapItem>
#include <QFile>

// 包含 QtCharts，ui 文件是不能直接编辑的，ui 文件生成的代码会用到 QtCharts
// 而 mainwindow 对应的 ui 文件生成的代码会包含头文件 mainwindow.h
// 因此可以将 QtCharts 的头文件包含写在这里
#include <QtCharts>
// 下面两句是等价的
// using namespace QtCharts
QT_CHARTS_USE_NAMESPACE


#include <armadillo>
using namespace arma;
#define ROW 200
#define COL 2

#define cimg_display 0
#include <CImg.h>
using namespace cimg_library;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_action_Quit_triggered();

    void on_action_Open_triggered();

    void on_actionGLCM_triggered();

    void glcm(const int &rowStep, const int &colStep, const int &grayLevel);

    void on_actionFractal_dimension_triggered();

    void on_actionFisher_triggered();

    void on_actionPerception_triggered();

    void on_actionMinimum_Distance_Classifier_triggered();

    void knn(const mat &data, const double &trainingRate, const int &k, const uvec &label,
             uvec &predictedLabel);

    void on_actionKNN_triggered();

private:
    Ui::MainWindow *ui;

    QString imageFormat = tr("All Images (*.bmp *.cur *.gif *.icns *.ico *.jp2 *.jpeg *.jpg *.mng *.pbm *.pgm *.png *.ppm *.svg *.svgz *.tga *.tif *.tiff *.wbmp *.webp *.xbm *.xpm);;");
    QString fileName;

    DialogGLCM *dlgGLCM;

    void setFileName(const QString &fileName);

    void setSaveFileName(const QString &saveFileName);

    inline int rgbToGray(const int &r, const int &g, const int &b);

    template <typename T>
    CImg<T> rgbToGray(const CImg<T> &img);

    template <typename T>
    inline bool isGrayscale(const CImg<T> &img);

    mat graycomatrix(const Mat<int> &SI, const int &grayLevel, const int &rowStep, const int &colStep);

    double getASM(const mat &glcmMatrix);

    double getIDM(const mat &glcmMatrix);

    double getContrast(const mat &glcmMatrix);

    double getCorrelation(const mat &glcmMatrix);

    double getEntropy(const mat &glcmMatrix);

    template <typename T>
    arma::Mat<T> cimgToMat(const CImg<T> &img);

    template <typename T>
    double boxcount(const arma::Mat<T> &img);

    template <typename T>
    CImg<T> padImage(const CImg<T> &img);

    void fisherTrain(const mat &data, const ivec &label, vec &weight, vec &dataProj, double &threshold);

    void fisherTest(const mat &data, const vec &weight, const double &threshold, const ivec &label,
                       ivec &predictedLabel, double &precision, double &recall, double &accuracy, double &F1);

    void perceptionTrain(const mat &data, const ivec &label, const double &learningRate, const int &maxEpoch, vec &weight);

    void perceptionTest(const mat &data, const ivec &label, const vec &weight,
                        ivec &predictedLabel, double &precision, double &recall, double &accuracy, double &F1);

    void minDistanceClassifier(const mat &data, const uvec &label, const double &trainRate, const unsigned int nCount,
                               uvec &predictedLabel, double &accuracy);

    template <typename T>
    Col<T> QVectorToCol(const QVector<T> &vector);

    template <typename T>
    int sign(const T &x);

    uword mode(const vec &v);
};

#endif // MAINWINDOW_H
