#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "dialogglcm.h"
#include <QMainWindow>
#include <QGraphicsScene>
#include <QPixmap>
#include <QGraphicsPixmapItem>
#include <QFile>

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

    void on_action_Save_triggered();

    void on_actionSave_as_triggered();

    void on_actionClose_triggered();

    void on_actionGLCM_triggered();

    void glcm(const int &distance, const int &theta, const int &grayLevel);

    void on_actionFractal_dimension_triggered();

    void on_actionFisher_triggered();

private:
    Ui::MainWindow *ui;

    QGraphicsScene *inScene, *outScene;
    QPixmap *inPixmap, *outPixmap;
    QGraphicsPixmapItem *inPixmapItem, *outPixmapItem;
    QString imageFormat = tr("All Images (*.bmp *.cur *.gif *.icns *.ico *.jp2 *.jpeg *.jpg *.mng *.pbm *.pgm *.png *.ppm *.svg *.svgz *.tga *.tif *.tiff *.wbmp *.webp *.xbm *.xpm);;");
    QString fileName;
    QString saveFileName;
    // result file name used for update out scene
    QString resultFileName;

    DialogGLCM *dlgGLCM;

    void cleanImage(void);
    void setFileName(const QString &fileName);
    void setSaveFileName(const QString &saveFileName);
    void updateOutScene(const QString &fileName);
    inline int rgbToGray(const int &r, const int &g, const int &b);
    template <typename T>
    CImg<T> rgbToGray(const CImg<T> &img);
    template <typename T>
    inline bool isGrayscale(const CImg<T> &img);
    template <typename T>
    CImg<T> getGlcm(const CImg<T> &img, const int &distance, const int &theta, const int &grayLevel);
    template <typename T>
    double getASM(const CImg<T> &img);
    template <typename T>
    double getIDM(const CImg<T> &img);
    template <typename T>
    double getContrast(const CImg<T> &img);
    template <typename T>
    double getCorrelation(const CImg<T> &img);
    template <typename T>
    double getEntropy(const CImg<T> &img);
    template <typename T>
    arma::Mat<T> cimgToMat(const CImg<T> &img);
    template <typename T>
    double boxcount(const arma::Mat<T> &img);
    template <typename T>
    CImg<T> padImage(const CImg<T> &img);

    void fisherTrain(const mat &data, const uvec &label, vec &weight, vec &dataProj, double &threshold);

    void fisherTesting(const mat &data, const vec &weight, const double &threshold, const uvec &label,
                       uvec &predictedLabel, double &precision, double &recall, double &accuracy, double &F1);

    void readCsv(const QString &fileName, QVector<double> &data);

    template <typename T>
    Col<T> QVectorToCol(const QVector<T> &vector);
};

#endif // MAINWINDOW_H
