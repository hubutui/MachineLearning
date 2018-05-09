#include "dialograndomdata2.h"
#include "ui_dialograndomdata2.h"

#include <QMessageBox>

DialogRandomData2::DialogRandomData2(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DialogRandomData2)
{
    ui->setupUi(this);
}

DialogRandomData2::~DialogRandomData2()
{
    delete ui;
}

// 以下四个槽函数都是为了保证协方差矩阵为对称矩阵
void DialogRandomData2::on_doubleSpinBox1Covar01_valueChanged(double arg1)
{
    ui->doubleSpinBox1Covar10->setValue(arg1);
}

void DialogRandomData2::on_doubleSpinBox1Covar10_valueChanged(double arg1)
{
    ui->doubleSpinBox1Covar01->setValue(arg1);
}

void DialogRandomData2::on_doubleSpinBox2Covar01_valueChanged(double arg1)
{
    ui->doubleSpinBox2Covar10->setValue(arg1);
}

void DialogRandomData2::on_doubleSpinBox2Covar10_valueChanged(double arg1)
{
    ui->doubleSpinBox2Covar01->setValue(arg1);
}

void DialogRandomData2::on_buttonBox_accepted()
{
    // 读取生成随机样本所需的信息，即样本数量、均值和协方差矩阵
    int N1 = ui->spinBoxN1->value();
    int N2 = ui->spinBoxN2->value();
    vec mu1 = {ui->doubleSpinBox1Mux->value(), ui->doubleSpinBox1Mux->value()};
    vec mu2 = {ui->doubleSpinBox2Mux->value(), ui->doubleSpinBox2Mux->value()};
    mat covariance1 = {{ui->doubleSpinBox1Covar00->value(), ui->doubleSpinBox1Covar01->value()},
                        {ui->doubleSpinBox1Covar10->value(), ui->doubleSpinBox1Covar11->value()}};
    mat covariance2 = {{ui->doubleSpinBox2Covar00->value(), ui->doubleSpinBox2Covar01->value()},
                        {ui->doubleSpinBox2Covar10->value(), ui->doubleSpinBox2Covar11->value()}};

    // 检查协方差矩阵是否为对称半正定矩阵
    if (det(covariance1) < 0 || det(covariance2) < 0) {
        QMessageBox::critical(this,
                              tr("Error"),
                              tr("Covariance must be symmetric positive semi-definite!"));
        return;
    }

    // 将这些参数作为信号参数发送出去
    emit sendData(N1, mu1, covariance1, N2, mu2, covariance2);
}
