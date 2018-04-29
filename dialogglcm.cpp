#include "dialogglcm.h"
#include "ui_dialogglcm.h"

DialogGLCM::DialogGLCM(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DialogGLCM)
{
    ui->setupUi(this);
}

DialogGLCM::~DialogGLCM()
{
    delete ui;
}

void DialogGLCM::on_buttonBox_accepted()
{
    emit sendData(ui->spinBoxRowStep->value(), ui->spinBoxColStep->value(), pow(2, ui->comboBoxGrayLevel->currentIndex()) + 1);
}

// 计算整数的指数函数
int DialogGLCM::pow(const int &base, const int &exponent)
{
    int result = 1;

    for (int i = 0; i < exponent; ++i) {
        result *= base;
    }

    return result;
}

// 设置步长的取值范围
// 显然它的最大值为 grayLevel - 1
// 这里主要是需要根据灰度级的变化来修改
// 最小值为 0
void DialogGLCM::on_comboBoxGrayLevel_currentIndexChanged(int index)
{
    ui->spinBoxRowStep->setMaximum(pow(2, index+1) - 1);
    ui->spinBoxColStep->setMaximum(pow(2, index+1) - 1);
}
