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

// explicação dos algoritmos
// marr-hildreth:
// também conhecido como laplacian of gaussian (LoG)
// ele combina suavização de bordas baseada na segunda derivada da intensidade da imagem..
// primeiramente, ele aplica o método gaussiano para reduzir ruídos.
// depois, ele aplica o laplace (segunda derivada)
// ele detecta bordas em todas as direções. de forma isotrópica.
// pode gerar "loops" em regiões com gradientes suaves.
// canny:
// é conhecido como o padrão-ouro para detectar bordas.
// calcula gradientes usando operadores (sobel)
// mantém apenas os pixels onde tem a magnitude máxima.

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

		varBetween := wB * wF * math.Pow(mB-mF, 2)
		if varBetween > varMax {
			varMax = varBetween
			threshold = uint8(t)
		}
	}

	newImg := image.NewGray(img.Bounds())
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			if img.GrayAt(x, y).Y > threshold {
				newImg.SetGray(x, y, color.Gray{255})
			} else {
				newImg.SetGray(x, y, color.Gray{0})
			}
		}
	}

	return newImg
}

func marrHildreth(img *image.Gray) *image.Gray {
	laplacianKernel := [][]float64{
		{0, 1, 0},
		{1, -4, 1},
		{0, 1, 0},
	}
	return applyConvolution(img, laplacianKernel, 1)
}

func watershed(img *image.Gray, bgPercentage float64) *image.Gray {
	if bgPercentage < 0 || bgPercentage > 1 {
		panic("bgPercentage deve estar entre 0 e 1")
	}

	var histogram [256]int
	totalPixels := img.Bounds().Dx() * img.Bounds().Dy()

	for y := 0; y < img.Bounds().Dy(); y++ {
		for x := 0; x < img.Bounds().Dx(); x++ {
			grayValue := img.GrayAt(x, y).Y
			histogram[grayValue]++
		}
	}

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

	inverted := image.NewGray(img.Bounds())

	for y := 0; y < img.Bounds().Dy(); y++ {
		for x := 0; x < img.Bounds().Dx(); x++ {
			if img.GrayAt(x, y).Y >= uint8(bgThreshold) {
				inverted.SetGray(x, y, color.Gray{0})
			} else {
				inverted.SetGray(x, y, color.Gray{255})
			}
		}
	}

	return inverted
}

// questao 3
func countObjects(img *image.Gray) int {
	smoothImg := image.NewGray(img.Bounds())
	for x := 1; x < img.Bounds().Dx()-1; x++ {
		for y := 1; y < img.Bounds().Dy()-1; y++ {
			var sum int
			count := 0
			for i := -1; i <= 1; i++ {
				for j := -1; j <= 1; j++ {
					sum += int(img.GrayAt(x+i, y+j).Y)
					count++
				}
			}
			smoothImg.SetGray(x, y, color.Gray{uint8(sum / count)})
		}
	}

	kernel := [][]int{
		{1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1},
	}

	erode := func(src *image.Gray) *image.Gray {
		result := image.NewGray(src.Bounds())
		offset := len(kernel) / 2
		for x := offset; x < src.Bounds().Dx()-offset; x++ {
			for y := offset; y < src.Bounds().Dy()-offset; y++ {
				fits := true
				for i := -offset; i <= offset && fits; i++ {
					for j := -offset; j <= offset && fits; j++ {
						if kernel[i+offset][j+offset] == 1 && src.GrayAt(x+i, y+j).Y != 0 {
							fits = false
						}
					}
				}
				if fits {
					result.SetGray(x, y, color.Gray{0})
				} else {
					result.SetGray(x, y, color.Gray{255})
				}
			}
		}
		return result
	}

	dilate := func(src *image.Gray) *image.Gray {
		result := image.NewGray(src.Bounds())
		offset := len(kernel) / 2
		for x := offset; x < src.Bounds().Dx()-offset; x++ {
			for y := offset; y < src.Bounds().Dy()-offset; y++ {
				hasBlack := false
				for i := -offset; i <= offset && !hasBlack; i++ {
					for j := -offset; j <= offset && !hasBlack; j++ {
						if kernel[i+offset][j+offset] == 1 && src.GrayAt(x+i, y+j).Y == 0 {
							hasBlack = true
						}
					}
				}
				if hasBlack {
					result.SetGray(x, y, color.Gray{0})
				} else {
					result.SetGray(x, y, color.Gray{255})
				}
			}
		}
		return result
	}

	temp := erode(smoothImg)
	eroded := erode(temp)
	temp = dilate(eroded)
	temp = dilate(temp)
	opened := dilate(temp)

	temp = dilate(opened)
	temp = dilate(temp)
	dilated := dilate(temp)
	temp = erode(dilated)
	temp = erode(temp)
	closed := erode(temp)

	width, height := closed.Bounds().Dx(), closed.Bounds().Dy()
	visited := make([][]bool, height)
	for i := range visited {
		visited[i] = make([]bool, width)
	}

	var directions = [][2]int{
		{-1, 0}, {1, 0}, {0, -1}, {0, 1},
		{-1, -1}, {-1, 1}, {1, -1}, {1, 1},
	}

	const minArea = 10
	var count int
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			if visited[y][x] || closed.GrayAt(x, y).Y == 255 {
				continue
			}

			area := 0
			stack := [][2]int{{x, y}}

			for len(stack) > 0 {
				px, py := stack[len(stack)-1][0], stack[len(stack)-1][1]
				stack = stack[:len(stack)-1]

				if visited[py][px] {
					continue
				}

				visited[py][px] = true
				area++

				for _, d := range directions {
					nx, ny := px+d[0], py+d[1]
					if nx >= 0 && ny >= 0 && nx < width && ny < height {
						if !visited[ny][nx] && closed.GrayAt(nx, ny).Y == 0 {
							stack = append(stack, [2]int{nx, ny})
						}
					}
				}
			}

			if area >= minArea {
				count++
			}
		}
	}

	return count
}

// QUESTAO CADEIA DE FREEMAN
func freemanChainCode(img *image.Gray) string {
	width, height := img.Bounds().Dx(), img.Bounds().Dy()
	visited := make([][]bool, height)
	for i := range visited {
		visited[i] = make([]bool, width)
	}

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

	var chain []int
	currentX, currentY := startX, startY
	visited[currentY][currentX] = true

	prevDir := 0

	for {
		nextDir := -1
		nextX, nextY := 0, 0

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

		if nextDir == -1 {
			break
		}

		chain = append(chain, nextDir)
		visited[nextY][nextX] = true
		currentX, currentY = nextX, nextY
		prevDir = (nextDir + 4) % 8 // Direção oposta para manter continuidade
	}

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
		return uint8(sum / count)
	}

	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
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

	watershedImg := watershed(img, 0.7)
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
