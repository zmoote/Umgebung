#include "mainwindow.hpp"
#include "umgebung/engine.hpp"
#include <QWidget>
#include <QHBoxLayout>
#include <QCoreApplication>

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("Umgebung Reality Engine - Control Panel");
    resize(400, 200);

    QWidget* centralWidget = new QWidget(this);
    QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);

    QHBoxLayout* controlLayout = new QHBoxLayout();
    QLabel* levelLabel = new QLabel("Simulation Levels:", this);
    levelsSpinBox = new QSpinBox(this);
    levelsSpinBox->setRange(1, 1000);
    levelsSpinBox->setValue(10);
    
    generateButton = new QPushButton("Generate PSUs", this);
    
    controlLayout->addWidget(levelLabel);
    controlLayout->addWidget(levelsSpinBox);
    controlLayout->addWidget(generateButton);

    statusLabel = new QLabel("Ready.", this);
    statusLabel->setAlignment(Qt::AlignCenter);

    mainLayout->addLayout(controlLayout);
    mainLayout->addWidget(statusLabel);

    setCentralWidget(centralWidget);

    connect(generateButton, &QPushButton::clicked, this, &MainWindow::handleGenerate);
}

void MainWindow::handleGenerate() {
    int levels = levelsSpinBox->value();
    
    statusLabel->setText("Generating...");
    // Force UI update before blocking (since we aren't using threads yet)
    QCoreApplication::processEvents();

    umgebung::FlowerOfLife engine;
    engine.generate(levels);

    const auto& units = engine.getUnits();
    QString status = QString("Success! Generated %1 PSUs.\nCUDA kernel execution complete.").arg(units.size());
    statusLabel->setText(status);
}
