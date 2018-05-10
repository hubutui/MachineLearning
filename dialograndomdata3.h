#ifndef DIALOGRANDOMDATA3_H
#define DIALOGRANDOMDATA3_H

#include <QDialog>

#include <armadillo>
using namespace arma;

namespace Ui {
class DialogRandomData3;
}

class DialogRandomData3 : public QDialog
{
    Q_OBJECT

public:
    explicit DialogRandomData3(QWidget *parent = 0);
    ~DialogRandomData3();

signals:
    void sendData(int N1,
                  vec mu1,
                  mat covariance1,
                  int N2,
                  vec mu2,
                  mat covariance2,
                  int N3,
                  vec mu3,
                  mat covariance3);

private slots:
    void on_doubleSpinBox1Covar01_valueChanged(double arg1);

    void on_doubleSpinBox1Covar10_valueChanged(double arg1);

    void on_doubleSpinBox2Covar01_valueChanged(double arg1);

    void on_doubleSpinBox2Covar10_valueChanged(double arg1);

    void on_doubleSpinBox3Covar01_valueChanged(double arg1);

    void on_doubleSpinBox3Covar10_valueChanged(double arg1);

    void on_buttonBox_accepted();

private:
    Ui::DialogRandomData3 *ui;
};

#endif // DIALOGRANDOMDATA3_H
