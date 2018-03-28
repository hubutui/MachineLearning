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
    int theta, grayLevel;

    if (ui->radioButtonD0->isChecked()) {
        theta = 0;
    } else if (ui->radioButtonD45->isChecked()) {
        theta = 45;
    } else if (ui->radioButtonD90->isChecked()) {
        theta = 90;
    } else if (ui->radioButtonD135->isChecked()) {
        theta = 135;
    } else {
        return;
    }

    if (ui->radioButtonL8->isChecked()) {
        grayLevel = 8;
    } else if (ui->radioButtonL16->isChecked()) {
        grayLevel = 16;
    } else if (ui->radioButtonL32->isChecked()) {
        grayLevel = 32;
    } else if (ui->radioButtonL64->isChecked()) {
        grayLevel = 64;
    } else if (ui->radioButtonL128->isChecked()) {
        grayLevel = 128;
    } else if (ui->radioButtonL256->isChecked()) {
        grayLevel = 256;
    } else {
        return;
    }
    emit sendData(ui->spinBoxDistance->value(), theta, grayLevel);
}
