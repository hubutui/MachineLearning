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
    void sendData(const int &distance, const int &theta, const int &grayLevel);

private slots:
    void on_buttonBox_accepted();

private:
    Ui::DialogGLCM *ui;
};

#endif // DIALOGGLCM_H
