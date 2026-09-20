#pragma once
#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
private slots:
    void handleGenerate();
private:
    QSpinBox* levelsSpinBox;
    QLabel* statusLabel;
    QPushButton* generateButton;
};
