package main

import (
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"log"
	"math"
	"os"
)

// Carregar imagem PNG em grayscale
func loadImage(filename string) *image.Gray {
	file, err := os.Open(filename)
	if err != nil {
		fmt.Println("Erro ao abrir a imagem!")
		log.Fatal(err)
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		log.Fatalf("Erro ao decodificar a imagem: %v", err)
	}

	gray := image.NewGray(img.Bounds())
	for x := 0; x < img.Bounds().Dx(); x++ {
		for y := 0; y < img.Bounds().Dy(); y++ {
			gray.Set(x, y, img.At(x, y))
		}
	}

	return gray
}

// Função para salvar uma imagem PNG
func saveImage(path string, img image.Image) {
	file, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	err = png.Encode(file, img)
	if err != nil {
		log.Fatal(err)
	}
}

// Aplica um kernel de convolução
func applyConvolution(img *image.Gray, kernel [][]float64, normalize float64) *image.Gray {
	width, height := img.Bounds().Dx(), img.Bounds().Dy()
	newImg := image.NewGray(img.Bounds())

	offset := len(kernel) / 2
	for x := offset; x < width-offset; x++ {
		for y := offset; y < height-offset; y++ {
			var sum float64
			for i := -offset; i <= offset; i++ {
				for j := -offset; j <= offset; j++ {
					sum += float64(img.GrayAt(x+i, y+j).Y) * kernel[i+offset][j+offset]
				}
			}
			newImg.SetGray(x, y, color.Gray{uint8(math.Min(255, sum/normalize))})
		}
	}

	return newImg
}

// Canny Edge Detection (usando Sobel)
func cannyEdgeDetection(img *image.Gray) *image.Gray {
	sobelX := [][]float64{
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1},
	}
	sobelY := [][]float64{
		{-1, -2, -1},
		{0, 0, 0},
		{1, 2, 1},
	}

	width, height := img.Bounds().Dx(), img.Bounds().Dy()
	newImg := image.NewGray(img.Bounds())

	for x := 1; x < width-1; x++ {
		for y := 1; y < height-1; y++ {
			var gx, gy float64
			for i := -1; i <= 1; i++ {
				for j := -1; j <= 1; j++ {
					gray := float64(img.GrayAt(x+i, y+j).Y)
					gx += gray * sobelX[i+1][j+1]
					gy += gray * sobelY[i+1][j+1]
				}
			}
			magnitude := math.Sqrt(gx*gx + gy*gy)
			newImg.SetGray(x, y, color.Gray{uint8(math.Min(255, magnitude))})
		}
	}

	return newImg
}
func otsuThreshold(img *image.Gray) *image.Gray {
	histogram := make([]int, 256)
	width, height := img.Bounds().Dx(), img.Bounds().Dy()
	totalPixels := width * height

	// Construindo o histograma
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			grayValue := img.GrayAt(x, y).Y
			histogram[grayValue]++
		}
	}

	var sum, sumB, wB, wF, varMax float64
	for i := 0; i < 256; i++ {
		sum += float64(i * histogram[i])
	}

	var threshold uint8
	for t := 0; t < 256; t++ {
		wB += float64(histogram[t])
		if wB == 0 {
			continue
		}
		wF = float64(totalPixels) - wB
		if wF == 0 {
			break
		}

		sumB += float64(t * histogram[t])
		mB := sumB / wB
		mF := (sum - sumB) / wF

		// Cálculo da variância entre classes
		varBetween := wB * wF * math.Pow(mB-mF, 2)
		if varBetween > varMax {
			varMax = varBetween
			threshold = uint8(t)
		}
	}

	// Aplicando limiarização na imagem com o threshold encontrado
	newImg := image.NewGray(img.Bounds())
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			if img.GrayAt(x, y).Y > threshold {
				newImg.SetGray(x, y, color.Gray{255}) // Branco
			} else {
				newImg.SetGray(x, y, color.Gray{0}) // Preto
			}
		}
	}

	return newImg
}

// Marr-Hildreth (Laplaciano do Gaussiano)
func marrHildreth(img *image.Gray) *image.Gray {
	laplacianKernel := [][]float64{
		{0, 1, 0},
		{1, -1, 1},
		{0, 1, 0},
	}
	return applyConvolution(img, laplacianKernel, 1)
}

// Watershed simplificado
func watershed(img *image.Gray, bgPercentage float64) *image.Gray {
	if bgPercentage < 0 || bgPercentage > 1 {
		panic("bgPercentage deve estar entre 0 e 1")
	}

	// Criar um histograma dos tons de cinza da imagem
	var histogram [256]int
	totalPixels := img.Bounds().Dx() * img.Bounds().Dy()

	for y := 0; y < img.Bounds().Dy(); y++ {
		for x := 0; x < img.Bounds().Dx(); x++ {
			grayValue := img.GrayAt(x, y).Y
			histogram[grayValue]++
		}
	}

	// Determinar o novo limiar com base na porcentagem
	targetPixels := int(float64(totalPixels) * bgPercentage)
	sum := 0
	bgThreshold := 0

	for i := 0; i < 256; i++ {
		sum += histogram[i]
		if sum >= targetPixels {
			bgThreshold = i
			break
		}
	}

	// Criar a imagem invertida
	inverted := image.NewGray(img.Bounds())

	for y := 0; y < img.Bounds().Dy(); y++ {
		for x := 0; x < img.Bounds().Dx(); x++ {
			if img.GrayAt(x, y).Y >= uint8(bgThreshold) {
				inverted.SetGray(x, y, color.Gray{0}) // Fundo
			} else {
				inverted.SetGray(x, y, color.Gray{255}) // Primeiro plano
			}
		}
	}

	return inverted
}

// questao 3
func countObjects(img *image.Gray) int {
	width, height := img.Bounds().Dx(), img.Bounds().Dy()
	visited := make([][]bool, height)

	for i := range visited {
		visited[i] = make([]bool, width)
	}

	var directions = [][2]int{
		{-1, 0}, {1, 0}, {0, -1}, {0, 1}, // Cima, Baixo, Esquerda, Direita
		{-1, -1}, {-1, 1}, {1, -1}, {1, 1}, // Diagonais
	}

	var count int
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// Se já foi visitado ou é fundo branco, ignora
			if visited[y][x] || img.GrayAt(x, y).Y == 255 {
				continue
			}

			// Encontramos um novo objeto
			count++
			stack := [][2]int{{x, y}} // Pilha para DFS

			// Flood Fill (DFS) para marcar todo o objeto como visitado
			for len(stack) > 0 {
				px, py := stack[len(stack)-1][0], stack[len(stack)-1][1]
				stack = stack[:len(stack)-1] // Pop

				// Se já foi visitado, pula
				if visited[py][px] {
					continue
				}

				visited[py][px] = true

				// Verifica todos os pixels vizinhos
				for _, d := range directions {
					nx, ny := px+d[0], py+d[1]

					// Se dentro dos limites e não visitado, adiciona na pilha
					if nx >= 0 && ny >= 0 && nx < width && ny < height {
						if !visited[ny][nx] && img.GrayAt(nx, ny).Y == 0 { // Apenas pixels pretos
							stack = append(stack, [2]int{nx, ny})
						}
					}
				}
			}
		}
	}

	return count
}

// QUESTAO CADEIA DE FREEMAN
// freemanChainCode gera o código de cadeia de Freeman para o primeiro objeto encontrado na imagem
func freemanChainCode(img *image.Gray) string {
	width, height := img.Bounds().Dx(), img.Bounds().Dy()
	visited := make([][]bool, height)
	for i := range visited {
		visited[i] = make([]bool, width)
	}

	// Direções do código de Freeman (0 a 7, sentido horário, começando da direita)
	directions := [][2]int{
		{1, 0},   // 0: Direita
		{1, -1},  // 1: Diagonal superior direita
		{0, -1},  // 2: Cima
		{-1, -1}, // 3: Diagonal superior esquerda
		{-1, 0},  // 4: Esquerda
		{-1, 1},  // 5: Diagonal inferior esquerda
		{0, 1},   // 6: Baixo
		{1, 1},   // 7: Diagonal inferior direita
	}

	// Encontrar o ponto inicial (primeiro pixel preto)
	var startX, startY int
	found := false
	for y := 0; y < height && !found; y++ {
		for x := 0; x < width && !found; x++ {
			if img.GrayAt(x, y).Y == 0 { // Pixel preto
				startX, startY = x, y
				found = true
			}
		}
	}

	if !found {
		return "Nenhum objeto encontrado"
	}

	// Armazenar o código de cadeia
	var chain []int
	currentX, currentY := startX, startY
	visited[currentY][currentX] = true

	// Direção inicial (arbitrariamente começamos olhando para a direita)
	prevDir := 0

	for {
		nextDir := -1
		nextX, nextY := 0, 0

		// Verificar as 8 direções a partir da direção anterior
		for i := 0; i < 8; i++ {
			dir := (prevDir + i) % 8 // Explorar em ordem a partir da direção anterior
			nx := currentX + directions[dir][0]
			ny := currentY + directions[dir][1]

			if nx >= 0 && nx < width && ny >= 0 && ny < height && !visited[ny][nx] && img.GrayAt(nx, ny).Y == 0 {
				nextDir = dir
				nextX, nextY = nx, ny
				break
			}
		}

		// Se não encontrou próximo pixel, contorno concluído
		if nextDir == -1 {
			break
		}

		chain = append(chain, nextDir)
		visited[nextY][nextX] = true
		currentX, currentY = nextX, nextY
		prevDir = (nextDir + 4) % 8 // Direção oposta para manter continuidade
	}

	// Converter o código de cadeia para string
	chainStr := ""
	for _, dir := range chain {
		chainStr += fmt.Sprintf("%d", dir)
	}

	return chainStr
}

// QUESTAO FILTRO BOX
func applyBoxFilter(img image.Image, size int) image.Image {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	filteredImg := image.NewGray(bounds)

	// Função auxiliar para calcular a média de uma janela
	average := func(x, y, size int) uint8 {
		var sum int
		var count int
		halfSize := size / 2
		for i := -halfSize; i <= halfSize; i++ {
			for j := -halfSize; j <= halfSize; j++ {
				nx, ny := x+i, y+j
				if nx >= 0 && nx < width && ny >= 0 && ny < height {
					r, _, _, _ := img.At(nx, ny).RGBA()
					// Convertendo para escala de cinza (simples média)
					gray := uint8((r + r + r) / 3)
					sum += int(gray)
					count++
				}
			}
		}
		// Retorna a média dos valores na janela
		return uint8(sum / count)
	}

	// Aplica o filtro box para cada pixel
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			// Calcular o valor médio para a janela de tamanho 'size' em (x, y)
			avg := average(x, y, size)
			filteredImg.Set(x, y, color.Gray{Y: avg})
		}
	}

	return filteredImg
}

// QUESTAO 6:
func segmentIntensity(img *image.Gray) *image.Gray {
	width, height := img.Bounds().Dx(), img.Bounds().Dy()
	segmented := image.NewGray(img.Bounds())

	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			grayValue := img.GrayAt(x, y).Y
			var newValue uint8

			// Aplicar a transformação conforme a tabela
			switch {
			case grayValue <= 50:
				newValue = 25
			case grayValue <= 100:
				newValue = 75
			case grayValue <= 150:
				newValue = 125
			case grayValue <= 200:
				newValue = 175
			default: // 201 a 255
				newValue = 255
			}

			segmented.SetGray(x, y, color.Gray{newValue})
		}
	}

	return segmented
}

func main() {
	path := os.Args[1]

	// var options int
	fmt.Println("Bem vindo ao Gotoshop!")
	img := loadImage(path)

	fmt.Println("Aplicando Canny...")
	canny := cannyEdgeDetection(img)
	saveImage("canny.png", canny)

	fmt.Println("Aplicando Otsu...")
	otsu := otsuThreshold(img)
	saveImage("otsu.png", otsu)

	fmt.Println("Aplicando Marr-Hildreth...")
	marr := marrHildreth(img)
	saveImage("marr_hildreth.png", marr)

	objectCount := countObjects(otsu)
	fmt.Printf("Número de objetos na imagem: %d\n", objectCount)

	fmt.Println("Aplicando Watershed...")

	watershedImg := watershed(img, 0.2)
	saveImage("watershed.png", watershedImg)

	fmt.Println("Processamento concluído! Imagens geradas:")
	fmt.Println("- canny.png")
	fmt.Println("- otsu.png")
	fmt.Println("- marr_hildreth.png")
	fmt.Println("- watershed.png")

	// Gerar o código de cadeia de Freeman
	chainCode := freemanChainCode(otsu)

	file, err := os.Create("freeman_chain.txt")
	if err != nil {
		log.Fatalf("Erro ao criar o arquivo: %v", err)
	}
	defer file.Close()

	_, err = file.WriteString(chainCode)
	if err != nil {
		log.Fatalf("Erro ao escrever no arquivo: %v", err)
	}

	fmt.Println("Código de cadeia salvo em freeman_chain.txt")

	// Aplicar os filtros Box 2x2, 3x3, 5x5, 7x7
	filtered2x2 := applyBoxFilter(img, 2)
	filtered3x3 := applyBoxFilter(img, 3)
	filtered5x5 := applyBoxFilter(img, 5)
	filtered7x7 := applyBoxFilter(img, 7)

	// Salvar as imagens filtradas
	saveImage("filtered_2x2.png", filtered2x2)
	saveImage("filtered_3x3.png", filtered3x3)
	saveImage("filtered_5x5.png", filtered5x5)
	saveImage("filtered_7x7.png", filtered7x7)

	// Indicar que o processamento foi concluído
	fmt.Println("Processamento concluído! Imagens geradas:")
	fmt.Println("- filtered_2x2.png")
	fmt.Println("- filtered_3x3.png")
	fmt.Println("- filtered_5x5.png")
	fmt.Println("- filtered_7x7.png")

	fmt.Println("Aplicando segmentação de intensidade...")
	segmentedImg := segmentIntensity(img)

	fmt.Println("Salvando a imagem segmentada...")
	saveImage("segmented.png", segmentedImg)

}
