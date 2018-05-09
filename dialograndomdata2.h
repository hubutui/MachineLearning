#ifndef DIALOGRANDOMDATA2_H
#define DIALOGRANDOMDATA2_H

#include <QDialog>
#include <armadillo>
using namespace arma;

namespace Ui {
class DialogRandomData2;
}

class DialogRandomData2 : public QDialog
{
    Q_OBJECT

public:
    explicit DialogRandomData2(QWidget *parent = 0);
    ~DialogRandomData2();

signals:
    void sendData(int N1, vec mu1, mat covariance1, int N2, vec mu2, mat covariance2);

private slots:

    void on_doubleSpinBox1Covar01_valueChanged(double arg1);

    void on_doubleSpinBox1Covar10_valueChanged(double arg1);

    void on_doubleSpinBox2Covar01_valueChanged(double arg1);

    void on_doubleSpinBox2Covar10_valueChanged(double arg1);

    void on_buttonBox_accepted();

private:
    Ui::DialogRandomData2 *ui;
};

#endif // DIALOGRANDOMDATA2_H
