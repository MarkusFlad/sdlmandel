#include <string>
#include <iostream>
#include <vector>
#include <complex>
#include <algorithm>
#include <thread>
#include <climits>
#if defined(__AVX512BW__) || defined(__AVX__) || defined(__SSE__)
#include <immintrin.h>
#endif
#include <SDL2/SDL.h>

// Put everything in a namespace forces inlining
namespace {

#ifdef DEBUG
const auto numberOfCpuCores = 1;
constexpr static std::size_t MAX_VECTORIZATION = 1;
#else
const auto numberOfCpuCores = std::thread::hardware_concurrency();
constexpr static std::size_t MAX_VECTORIZATION = 8;
#endif

// Class that encapsulates the SDL Surface. Instances of this class can be used
// to prepare an image in normal RAM (which the CPU can access).
// Provides interlaced canvases that allow different threads to update different
// areas of the image at the same time.
class Surface {
private:
    static constexpr bool isBigEndian =
            SDL_BYTEORDER == SDL_BIG_ENDIAN ? true : false;
    static constexpr Uint32 R_MASK = isBigEndian ? 0xff000000 : 0x000000ff;
    static constexpr Uint32 G_MASK = isBigEndian ? 0x00ff0000 : 0x0000ff00;
    static constexpr Uint32 B_MASK = isBigEndian ? 0x0000ff00 : 0x00ff0000;
    static constexpr Uint32 A_MASK = isBigEndian ? 0x000000ff : 0xff000000;
    static constexpr int COLOR_DEPTH_BYTES = 4;
    static constexpr int COLOR_DEPTH = CHAR_BIT * COLOR_DEPTH_BYTES;
public:
    typedef std::size_t Size;
public:
    Surface(int width, int height)
    : _surface(SDL_CreateRGBSurface(
            0, width, height, COLOR_DEPTH, R_MASK, G_MASK, B_MASK, A_MASK))
    , _pixelNumber (width * height) {
    }
    ~Surface() {
        if (_surface) {
            SDL_FreeSurface(_surface);
        }
    }
    Size width() const {
        return _surface->w;
    }
    Size height() const {
        return _surface->h;
    }
    Size pixelNumber() const {
        return _pixelNumber;
    }
    Uint32* pixels() {
        return static_cast<Uint32*>(_surface->pixels);
    }
    struct Line {
        Size y;
        Size width;
        Uint32* pixels;
    };
    // The InterlacedCanvas provides interlaced access to the Surface. Each
    // thread must use its own InterlacedCanvas to write to the bitmap.
    class InterlacedCanvas {
    public:
        class Iterator {
        public:
            Iterator(Size y, Size _width, Uint32* pixels,
                    Size interlaceIncrement, Size pixelPointerIncrement)
            : _il {y, _width, pixels}
            , _interlaceIncrement (interlaceIncrement)
            , _pixelPointerIncrement (pixelPointerIncrement) {
            }
            Line& operator*() {
                return _il;
            }
            bool operator!=(const Iterator& other) const {
                return _il.pixels != other._il.pixels;
            }
            Iterator& operator++() {
                _il.y += _interlaceIncrement;
                _il.pixels += _pixelPointerIncrement;
                return *this;
            }
        private:
            Line _il;
            Size _interlaceIncrement;
            Size _pixelPointerIncrement;
        };
        InterlacedCanvas(Surface& surface, Size yStart, Size increment)
        : _surface (surface)
        , _yStart (yStart)
        , _increment (increment)
        , _pixelsStart (yStart * surface.width())
        , _pixelsPointerIncrement (increment * surface.width()) {
        }
        Size width() const {
            return _surface.width();
        }
        Size height() const {
            return _surface.height();
        }
        Iterator begin() {
            return Iterator(_yStart, _surface.width(),
                    _surface.pixels() + _pixelsStart,
                    _increment, _pixelsPointerIncrement);
        }
        Iterator end() {
            return Iterator(_yStart + _surface.height(), _surface.width(),
                    _surface.pixels() + _surface.pixelNumber() + _pixelsStart,
                    _increment, _pixelsPointerIncrement);
        }
    private:
        Surface& _surface;
        Size _yStart;
        Size _increment;
        Size _pixelsStart;
        Size _pixelsPointerIncrement;
    };
    std::vector<InterlacedCanvas> provideInterlacedCanvas(Size increment) {
        std::vector<InterlacedCanvas> interlacedCanvasVector;
        for (Size yStart=0; yStart<increment; yStart++) {
            interlacedCanvasVector.emplace_back(*this, yStart, increment);
        }
        return interlacedCanvasVector;
    }
    static Size roundToMultiple (Size number, Size base) {
        return number + ((number % base) ? (base - number % base) : 0);
    }
    friend class Renderer;
private:
    SDL_Surface* _surface;
    Size _pixelNumber;
};

// Class that encapsulates a SDL texture. A Texture represents an image in the
// RAM of the graphics card. It cannot be manipulated directly by the CPU. A
// Texture must be created from a surface. The CPU can only directly access that
// Surface.
class Texture {
private:
    SDL_Texture* _texture;
public:
    Texture(SDL_Texture* _texture)
    : _texture(_texture) {
    }
    ~Texture() {
        if (_texture) {
            SDL_DestroyTexture(_texture);
        }
    }
    friend class Renderer;
};

// Class that encapsulates an SDL Context. A corresponding object must be
// created and be valid as long as SDL is required.
class SdlContext {
public:
    SdlContext() {
        SDL_Init(SDL_INIT_EVERYTHING);
    }
    ~SdlContext() {
        SDL_Quit();
    }
};

// Class that encapsulates an SDL window. Represents a Window on the screen.
class Window {
private:
    SDL_Window* _window;
public:
    Window(std::string title, int width, int height)
    : _window(SDL_CreateWindow(title.c_str(),
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            width, height, 0)) {
    }
    ~Window() {
        if (_window) {
            SDL_DestroyWindow(_window);
        }
    }
    friend class Renderer;
};

// Class that encapsulates an SDL renderer. Allows you to create textures from
// surfaces and copy textures to the target rendering graphics card.
class Renderer {
private:
    SDL_Renderer* _renderer;
public:
    Renderer(Window& window)
    : _renderer(SDL_CreateRenderer(
            window._window, -1, SDL_RENDERER_ACCELERATED)) {
    }
    SDL_Renderer* GetSdlObject() const {
        return _renderer;
    }
    SDL_Texture* CreateSdlTexture(Surface& surface) {
        return SDL_CreateTextureFromSurface(_renderer, surface._surface);
    }
    void Copy(Texture& texture) {
        SDL_RenderCopy(_renderer, texture._texture, NULL, NULL);
    }
    void Clear() {
         SDL_RenderClear(_renderer);
    }
    void Present() {
         SDL_RenderPresent(_renderer);
    }
    ~Renderer() {
        if (_renderer) {
            SDL_DestroyRenderer(_renderer);
        }
    }
};

// If the system does not support SIMD, NoSimdUnion can be used.
struct NoSimdUnion {
    typedef double NumberType;
    typedef double SimdRegisterType;
    NoSimdUnion()
    : reg(val) {
    }
    NoSimdUnion(const NoSimdUnion& other)
    : reg(val) {
        std::copy(std::begin(other.val), std::end(other.val), std::begin(val));
    }
    NoSimdUnion& operator=(const NoSimdUnion& other) {
        std::copy(std::begin(other.val), std::end(other.val), std::begin(val));
        return *this;
    }
    SimdRegisterType* reg;
    NumberType val[8];
};

#if defined(__AVX512BW__) || defined(__AVX__) || defined(__SSE__)
union Simd128DUnion {
    typedef double NumberType;
    typedef __m128d SimdRegisterType;
    SimdRegisterType reg[4];
    NumberType val[8];
};

union Simd256DUnion {
    typedef double NumberType;
    typedef __m256d SimdRegisterType;
    SimdRegisterType reg[2];
    NumberType val[8];
};

union Simd512DUnion {
    typedef double NumberType;
    typedef __m512d SimdRegisterType;
    SimdRegisterType reg[1];
    NumberType val[8];
};
#endif // defined(__AVX512BW__) || defined(__AVX__) || defined(__SSE__)

template<class SimdUnion>
constexpr std::size_t numberOfNumbers() {
    return sizeof(SimdUnion::val) / sizeof(typename SimdUnion::NumberType);
}
template<class SimdUnion>
constexpr std::size_t numberOfNumbersInRegister() {
    return sizeof(typename SimdUnion::SimdRegisterType) /
            sizeof(typename SimdUnion::NumberType);
}
template<class SimdUnion>
constexpr std::size_t numberOfRegisters() {
    return numberOfNumbers<SimdUnion>() /
            numberOfNumbersInRegister<SimdUnion>();
}
template<class SimdUnion>
void setValue(SimdUnion& simdUnion, typename SimdUnion::NumberType v) {
    typedef typename SimdUnion::SimdRegisterType SimdRegisterType;
    SimdRegisterType* vValues = simdUnion.reg;
    constexpr auto numbersInReg = numberOfNumbersInRegister<SimdUnion>();
    for (std::size_t i=0; i<numberOfNumbers<SimdUnion>(); i+=numbersInReg) {
        if constexpr (numbersInReg == 1) {
            *vValues = v;
        } else if constexpr (numbersInReg == 2) {
            *vValues = SimdRegisterType{v, v};
        } else if constexpr (numbersInReg == 4) {
            *vValues = SimdRegisterType{v, v, v, v};
        } else if constexpr (numbersInReg == 8) {
            *vValues = SimdRegisterType{v, v, v, v, v, v, v, v};
        }
        vValues++;
    }
}

// VectorizedComplex provides a convenient interface to deal with complex
// numbers and uses the power of SIMD for high execution speed.
template <class SimdUnion>
class VectorizedComplex {
public:
    typedef typename SimdUnion::NumberType NumberType;
    typedef typename SimdUnion::SimdRegisterType SimdRegisterType;
    typedef std::size_t Size;

    // SquaredAbs is passed to special VectorizedComplex methods that calculate
    // the squared absolute value of the complex number as an intermediate.
    // This means that the calculation does not have to be done twice.
    class SquaredAbs {
    public:
        void simdReg(Size i, const SimdRegisterType& reg) {
            _squaredAbs.reg[i] = reg;
        }
        bool operator>(NumberType threshold) const {
            const auto& sqrdAbsVals = _squaredAbs.val;
            return std::all_of(std::begin(sqrdAbsVals), std::end(sqrdAbsVals),
                    [&threshold](auto v) { return v>threshold; });
        }
        bool thresholdExceeded(Size i, NumberType threshold) {
            return (_squaredAbs.val[i] > threshold);
        }
    private:
        SimdUnion _squaredAbs;
    };
    VectorizedComplex() = default;
    VectorizedComplex(const VectorizedComplex&) = default;
    VectorizedComplex(const SimdUnion& reals, NumberType commonImagValue)
    : _reals(reals) {
        setValue(_imags, commonImagValue);
    }
    VectorizedComplex& square(SquaredAbs& squaredAbs) {
        for (Size i=0; i<numberOfRegisters<SimdUnion>(); i++) {
            auto realSquared = _reals.reg[i] * _reals.reg[i];
            auto imagSquared = _imags.reg[i] * _imags.reg[i];
            auto realTimesImag = _reals.reg[i] * _imags.reg[i];
            _reals.reg[i] = realSquared - imagSquared;
            _imags.reg[i] = realTimesImag + realTimesImag;
            squaredAbs.simdReg(i, realSquared + imagSquared);
        }
        return *this;
    }
    friend VectorizedComplex operator+(const VectorizedComplex& lhs,
            const VectorizedComplex& rhs) {
        VectorizedComplex resultNumbers;
        for (Size i=0; i<numberOfRegisters<SimdUnion>(); i++) {
            resultNumbers._reals.reg[i] = lhs._reals.reg[i] + rhs._reals.reg[i];
            resultNumbers._imags.reg[i] = lhs._imags.reg[i] + rhs._imags.reg[i];
        }
        return resultNumbers;
    }
private:
    SimdUnion _reals;
    SimdUnion _imags;
};

// Class that represents a section in the complex plane. This section is defined
// by a minimum and maximum complex number.
template <class NumberType>
class ComplexPlaneSection {
public:
    ComplexPlaneSection (const std::complex<NumberType>& min,
            const std::complex<NumberType>& max)
    : _min(min)
    , _max(max) {
    }
    std::complex<NumberType> min() const {
        return _min;
    }
    std::complex<NumberType> max() const {
        return _max;
    }
    NumberType realRange() const {
        return _max.real() - _min.real();
    }
    NumberType imagRange() const {
        return _max.imag() - _min.imag();
    }
    std::complex<NumberType> center() const {
        return std::complex<NumberType>(realRange() / 2.0, imagRange() / 2.0);
    }
    std::complex<NumberType> valueAtPixel(int x, int y, int width, int height) {
        NumberType onePixelInReal = realRange() / width;
        NumberType onePixelInImag = imagRange() / height;
        NumberType valueReal = _min.real() +
                static_cast<NumberType>(x) * onePixelInReal;
        NumberType valueImag = _min.imag() +
                static_cast<NumberType>(y) * onePixelInImag;
        return std::complex<NumberType>(valueReal, valueImag);
    }
private:
    std::complex<NumberType> _min;
    std::complex<NumberType> _max;
};

template<class NumberType>
ComplexPlaneSection<NumberType> complexPlaneSectionAroundCenter (
        NumberType realCenter, NumberType imagCenter, NumberType imagRange,
        int xMin, int yMin, int xMax, int yMax) {
    int yRange = yMax - yMin;
    NumberType pixelPerValue = static_cast<NumberType>(yRange) / imagRange;
    int xRange = xMax - xMin;
    NumberType realRange = static_cast<NumberType>(xRange) / pixelPerValue;
    NumberType realRangeHalf = realRange / 2.0;
    NumberType realMin = realCenter - realRangeHalf;
    NumberType realMax = realCenter + realRangeHalf;
    NumberType imagRangeHalf = imagRange / 2.0;
    NumberType imagMin = imagCenter - imagRangeHalf;
    NumberType imagMax = imagCenter + imagRangeHalf;
    return ComplexPlaneSection<NumberType>(
            std::complex<NumberType>(realMin, imagMin),
            std::complex<NumberType>(realMax, imagMax));
}

template<class NumberType>
ComplexPlaneSection<NumberType> complexPlaneSectionAroundCenter (
        NumberType realCenter, NumberType imagCenter, NumberType imagRange,
        int width, int height) {
    return complexPlaneSectionAroundCenter(realCenter, imagCenter, imagRange,
            0, 0, width, height);
}

template<class NumberType>
ComplexPlaneSection<NumberType> zoomedComplexPlane (
        const ComplexPlaneSection<NumberType>& originalCps,
        const std::complex<NumberType> newCenter, NumberType factor) {
    NumberType step = 1.0 - 1.0 / factor;
    NumberType newCenterRealDistanceToMin =
            newCenter.real() - originalCps.min().real();
    NumberType newCenterRealDistanceToMax =
            originalCps.max().real() - newCenter.real();
    NumberType cNewMinReal =
            originalCps.min().real() + newCenterRealDistanceToMin * step;
    NumberType cNewMaxReal =
            originalCps.max().real() - newCenterRealDistanceToMax * step;
    NumberType newCenterImagDistanceToMin =
            newCenter.imag() - originalCps.min().imag();
    NumberType newCenterImagDistanceToMax =
            originalCps.max().imag() - newCenter.imag();
    NumberType cNewMinImag =
            originalCps.min().imag() + newCenterImagDistanceToMin * step;
    NumberType cNewMaxImag =
            originalCps.max().imag() - newCenterImagDistanceToMax * step;
    return ComplexPlaneSection<NumberType>(
            std::complex<NumberType>(cNewMinReal, cNewMinImag),
            std::complex<NumberType>(cNewMaxReal, cNewMaxImag));
}

// The ComplexPlaneCalculator performs function f(c), with c as a
// VectorizedComplex and a byte as the return value. Due to its eightfold
// vectorization, each returned bit can return a Boolean value from the
// calculation f(c). The full byte is then written to the canvas. This is done
// until the whole bitmap is filled.
template <class SimdUnion, class Functor>
class ComplexPlaneCalculator {
public:
    typedef VectorizedComplex<SimdUnion> VComplex;
    typedef typename SimdUnion::NumberType NumberType;
    typedef typename Surface::Line Line;
    typedef std::size_t Size;

    ComplexPlaneCalculator(const ComplexPlaneSection<NumberType>& cps,
            Surface::InterlacedCanvas& canvas, Functor f)
    : _cps(cps)
    , _canvas(canvas)
    , _f(f) {
    }
    void operator()() {
        const NumberType realRange = _cps.realRange();
        const NumberType imagRange = _cps.imagRange();
        const NumberType rasterReal = realRange / _canvas.width();
        const NumberType rasterImag = imagRange / _canvas.height();
        std::vector<SimdUnion> cRealValues;
        cRealValues.reserve(_canvas.width() / MAX_VECTORIZATION);
        for (Size x=0; x<_canvas.width(); x+=MAX_VECTORIZATION) {
            SimdUnion cReals;
            for (Size i=0; i<MAX_VECTORIZATION; i++) {
                cReals.val[i] = _cps.min().real() + (x+i)*rasterReal;
            }
            cRealValues.push_back(cReals);
        }
        for (Line& line : _canvas) {
            Uint32* nextPixels = line.pixels;
            const NumberType cImagValue = _cps.min().imag() + line.y*rasterImag;
            for (const SimdUnion& cReals : cRealValues) {
                const VComplex c(cReals, cImagValue);
                std::array<Uint32, MAX_VECTORIZATION> colors (_f(c));
                std::copy(colors.begin(), colors.end(), nextPixels);
                nextPixels+=MAX_VECTORIZATION;
            }
        }
    }
private:
    ComplexPlaneSection<NumberType> _cps;
    Surface::InterlacedCanvas _canvas;
    Functor _f;
};

// Functor calculating a Mandelbrot iteration for a VectorizedComplex. This
// means that for eight complex numbers the Mandelbrot calculation is
// (potentially) executed in parallel. The result is a vector of colors that
// encode the number of iterations.
template <class SimdUnion, class ColorEncoder>
class MandelbrotFunction {
public:
    typedef VectorizedComplex<SimdUnion> VComplex;
    typedef typename SimdUnion::NumberType NumberType;
    typedef std::size_t Size;

    MandelbrotFunction(Size maxIterations, ColorEncoder colorEncoder,
            NumberType pointOfNoReturn = 2.0)
    : _maxIterations(maxIterations)
    , _colorEncoder(colorEncoder)
    , _squaredPointOfNoReturn(pointOfNoReturn * pointOfNoReturn) {
    }
    std::array<Uint32, MAX_VECTORIZATION> operator()(const VComplex& c) const {
        std::array<Size, MAX_VECTORIZATION> iterations;
        std::fill(iterations.begin(), iterations.end(), _maxIterations);
        VComplex z = c;
        typename VComplex::SquaredAbs squaredAbs;
        for (Size i=0; i<_maxIterations; i++) {
            z = z.square(squaredAbs) + c;
            std::size_t done = 0;
            for (Size j=0; j<MAX_VECTORIZATION; j++) {
                if (iterations[j] == _maxIterations && squaredAbs.
                        thresholdExceeded(j, _squaredPointOfNoReturn)) {
                    iterations[j] = i;
                    done++;
                }
            }
            if (done >= MAX_VECTORIZATION) {
                break;
            }
        }
        return _colorEncoder(iterations, _maxIterations);
    }
private:
    Size _maxIterations;
    ColorEncoder _colorEncoder;
    NumberType _squaredPointOfNoReturn;
};

// Functor calculating a set of 8 colors out of a given iteration number set.
class SimpleColorEncoder {
public:
    typedef std::size_t Size;

    std::array<Uint32, MAX_VECTORIZATION> operator()(
            const std::array<Size, MAX_VECTORIZATION>& iterations,
			Size maxIterations) const {
        std::array<Uint32, MAX_VECTORIZATION> colors;
        for (Size i=0; i<iterations.size(); i++) {
            Uint32 it = iterations[i];
            if (it < maxIterations) {
                Uint32 alpha = 0xff000000;
                Uint32 blue = ((it * 5) % 0x100) << 16;
                Uint32 green = ((it * 3) % 0x100) << 8;
                Uint32 red = ((it * 2) % 0x100);
                colors[i] = alpha | red | green | blue;
            } else {
                colors[i] = 0x00000000;
            }
        }
        return colors;
    }
};

#if defined(__AVX512BW__)
typedef Simd512DUnion SystemSimdUnion;
#elif defined __AVX__
typedef Simd256DUnion SystemSimdUnion;
#elif defined __SSE__
typedef Simd128DUnion SystemSimdUnion;
#else
typedef NoSimdUnion SystemSimdUnion;
#endif

} // end namespace

int main(int argc, char** argv) {
    SdlContext sdlContext;
    typedef SystemSimdUnion::NumberType NumberType;
    typedef ComplexPlaneCalculator<SystemSimdUnion,
            MandelbrotFunction<SystemSimdUnion,
            SimpleColorEncoder>> MandelbrotCalculator;
    std::size_t width = 1024;
    std::size_t height = 768;
    const NumberType iterationsFactor = 0.0025;
    NumberType maxIterations = 100.0;
    Surface surface(width, height);
    auto canvasVector = surface.provideInterlacedCanvas(numberOfCpuCores);
    Window window("Mandelbrot", width, height);
    Renderer renderer(window);
    SDL_Event input;
    ComplexPlaneSection<NumberType> cps =
    		complexPlaneSectionAroundCenter<NumberType>(
    				-0.5, 0.0, 2.0, width, height);
    std::complex<NumberType> cpsCenter = cps.center();
    const NumberType fastZoomFactor = 1.05;
    const NumberType slowZoomFactor = 1.01;
    NumberType zoomFactor = slowZoomFactor;
    bool quit = false;
    std::complex<NumberType> mousePos = cpsCenter.real();

    while(!quit) {
        std::vector<std::thread> threads;
        for (auto& canvas : canvasVector) {
            threads.emplace_back(MandelbrotCalculator (cps, canvas,
                    MandelbrotFunction<SystemSimdUnion,
                    SimpleColorEncoder> (maxIterations,
                            SimpleColorEncoder())));
        }
        for (auto& t : threads) {
            t.join();
        }
        Texture texture = renderer.CreateSdlTexture(surface);
        renderer.Clear();
        renderer.Copy(texture);
        renderer.Present();
        while (SDL_PollEvent(&input) > 0) {
            if (input.type == SDL_QUIT) {
                quit = true;
                break;
            } else if (input.type == SDL_MOUSEMOTION) {
                int x,y;
                SDL_GetMouseState(&x,&y);
                mousePos = cps.valueAtPixel(x, y, width ,height);
            } else if (input.type == SDL_MOUSEBUTTONDOWN) {
                if (input.button.button == SDL_BUTTON_LEFT) {
                    zoomFactor = fastZoomFactor;
                } else if (input.button.button == SDL_BUTTON_RIGHT) {
                    zoomFactor = 1.0 / fastZoomFactor;
                }
            } else if (input.type == SDL_MOUSEBUTTONUP) {
                if (input.button.button == SDL_BUTTON_LEFT) {
                    zoomFactor = slowZoomFactor;
                } else if (input.button.button == SDL_BUTTON_RIGHT) {
                    zoomFactor = 1.0 / slowZoomFactor;
                }
            }
        }
        cps = zoomedComplexPlane(cps, mousePos, zoomFactor);
        if (zoomFactor > 1.0) {
			maxIterations = maxIterations +
					maxIterations * (zoomFactor * iterationsFactor);
        } else if (zoomFactor < 1.0) {
			maxIterations = maxIterations -
					maxIterations * (zoomFactor * iterationsFactor);
        }
    }
    return 0;
}
