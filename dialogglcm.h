#ifndef DIALOGGLCM_H
#define DIALOGGLCM_H

#include <QDialog>

namespace Ui {
class DialogGLCM;
}

class DialogGLCM : public QDialog
{
    Q_OBJECT

public:
    explicit DialogGLCM(QWidget *parent = 0);
    ~DialogGLCM();

signals:
    void sendData(const int &rowStep, const int &colStep, const int &grayLevel);

private slots:
    void on_buttonBox_accepted();

    void on_comboBoxGrayLevel_currentIndexChanged(int index);

private:
    Ui::DialogGLCM *ui;
    inline int pow(const int &base, const int &exponent);
};

#endif // DIALOGGLCM_H
