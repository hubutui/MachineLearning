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

    inScene = new QGraphicsScene;
    outScene = new QGraphicsScene;

    ui->graphicsView_in->setScene(inScene);
    ui->graphicsView_out->setScene(outScene);

    inPixmap = new QPixmap;
    outPixmap = new QPixmap;
    inPixmapItem = inScene->addPixmap(*inPixmap);
    outPixmapItem = outScene->addPixmap(*outPixmap);

    resultFileName = "tmp.tiff";
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_action_Quit_triggered()
{
    QApplication::quit();
}

void MainWindow::cleanImage()
{
    inScene->clear();
    outScene->clear();
}

void MainWindow::setFileName(const QString &fileName)
{
    this->fileName = fileName;
}

void MainWindow::setSaveFileName(const QString &saveFileName)
{
    this->saveFileName = saveFileName;
}

void MainWindow::updateOutScene(const QString &fileName)
{
    qDebug() << "repaint out image." << endl;
    outScene->clear();
    outPixmap->load(fileName);
    outPixmapItem = outScene->addPixmap(*outPixmap);
    outScene->setSceneRect(QRectF(outPixmap->rect()));
}

void MainWindow::on_action_Open_triggered()
{
    QString imagePath = QFileDialog::getOpenFileName(
                this, tr("Open file"), QDir::homePath(), imageFormat);

    // check if file is valid
    if (!imagePath.isEmpty()) {
        QFile file(imagePath);

        if(!file.open(QIODevice::ReadOnly)) {
            QMessageBox::critical(this, tr("Error"), tr("Unable to read image!"));
            return;
        }

        // clear previouly showed image
        cleanImage();

        // show image
        inPixmap->load(imagePath);
        inPixmapItem = inScene->addPixmap(*inPixmap);
        inScene->setSceneRect(QRectF(inPixmap->rect()));

        // save fileName for later use
        setFileName(imagePath);
    }
}

void MainWindow::on_action_Save_triggered()
{
    if (!saveFileName.isEmpty()) {
        outPixmapItem->pixmap().save(saveFileName);
    } else {
        on_actionSave_as_triggered();
    }
}

void MainWindow::on_actionSave_as_triggered()
{
    QString savePath = QFileDialog::getSaveFileName(this, tr("Save image"), QDir::homePath(), imageFormat);

    if (!savePath.isEmpty()) {
        QFile file(savePath);

        if (!file.open(QIODevice::WriteOnly)) {
            QMessageBox::critical(this, tr("Error!"), tr("Unable to save image!"));
            return;
        }

        outPixmapItem->pixmap().save(savePath);
        setSaveFileName(savePath);
    }
}

void MainWindow::on_actionClose_triggered()
{
    cleanImage();
    setFileName("");
}
