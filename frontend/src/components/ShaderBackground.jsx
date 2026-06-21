import { useEffect, useRef } from "react";

const vertexShaderSource = `
attribute vec2 position;

void main() {
  gl_Position = vec4(position, 0.0, 1.0);
}
`;

const fragmentShaderSource = `
precision highp float;

uniform vec2 iResolution;
uniform vec2 iMouse;
uniform float iTime;

const float c = 8.0;

vec2 Scale(vec2 p) {
  return (p * 2.0 - iResolution.xy) / iResolution.y;
}

void main() {
  vec2 fragCoord = gl_FragCoord.xy;
  vec2 scaledp = Scale(fragCoord);
  vec2 mouse = Scale(iMouse);

  vec2 dir = normalize(vec2(
    sin(iTime * 0.5),
    cos(iTime * 0.5)
  ));

  float separation = 0.15;
  vec2 PlusPole = mouse + dir * separation;
  vec2 MinPole = mouse - dir * separation;

  float base = length(scaledp) * 0.3;
  vec3 col = vec3(base);

  vec2 blocked = mod(scaledp * c + 0.5, 1.0) * 2.0 - 1.0;
  vec2 middle = floor(scaledp * c + 0.5) / c;

  vec2 delta1 = PlusPole - middle;
  vec2 force1 = delta1 / dot(delta1, delta1);

  vec2 delta2 = MinPole - middle;
  vec2 force2 = -delta2 / dot(delta2, delta2);

  vec2 forcer = force1 + force2;

  float d = abs(-blocked.x * forcer.y + blocked.y * forcer.x) / length(forcer);
  float bd = length(blocked);

  float shade = 0.0;

  if (bd < 0.95) {
    if (d < 0.1 && d > 0.0) {
      float align = dot(normalize(forcer), normalize(blocked));
      shade = 0.28 + 0.2 * align;
    } else {
      shade = 0.1;
    }
  } else if (bd > 1.0) {
    shade = 0.025;
  } else {
    shade = 0.0;
  }

  float distToMouse = length(scaledp - mouse);
  shade += 0.045 / (distToMouse * 12.0 + 0.22);

  shade = smoothstep(0.0, 0.42, shade) * 0.55;
  shade = mix(shade, shade * 0.72, 0.65);

  gl_FragColor = vec4(vec3(shade), 1.0);
}
`;

function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(message || "Shader compilation failed.");
  }

  return shader;
}

export default function ShaderBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;

    const gl = canvas.getContext("webgl", { alpha: false, antialias: true });
    if (!gl) return undefined;

    let animationFrameId = 0;
    let disposed = false;
    const mouse = {
      x: window.innerWidth * 0.5,
      y: window.innerHeight * 0.5,
    };

    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
    const program = gl.createProgram();

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const message = gl.getProgramInfoLog(program);
      throw new Error(message || "Program linking failed.");
    }

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
      gl.STATIC_DRAW,
    );

    const positionLocation = gl.getAttribLocation(program, "position");
    const resolutionLocation = gl.getUniformLocation(program, "iResolution");
    const mouseLocation = gl.getUniformLocation(program, "iMouse");
    const timeLocation = gl.getUniformLocation(program, "iTime");

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.floor(window.innerWidth * dpr);
      const height = Math.floor(window.innerHeight * dpr);

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
      gl.viewport(0, 0, width, height);
    }

    function handlePointerMove(event) {
      mouse.x = event.clientX;
      mouse.y = window.innerHeight - event.clientY;
    }

    const start = performance.now();

    function render() {
      if (disposed) return;

      resize();
      gl.useProgram(program);

      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.enableVertexAttribArray(positionLocation);
      gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

      gl.uniform2f(resolutionLocation, canvas.width, canvas.height);
      gl.uniform2f(mouseLocation, mouse.x, mouse.y);
      gl.uniform1f(timeLocation, (performance.now() - start) / 1000);

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      animationFrameId = window.requestAnimationFrame(render);
    }

    resize();
    window.addEventListener("resize", resize);
    window.addEventListener("mousemove", handlePointerMove);
    render();

    return () => {
      disposed = true;
      window.cancelAnimationFrame(animationFrameId);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", handlePointerMove);
      gl.deleteBuffer(buffer);
      gl.deleteProgram(program);
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
    };
  }, []);

  return <canvas ref={canvasRef} className="pointer-events-none fixed inset-0 z-0 h-full w-full opacity-80 [filter:grayscale(1)_contrast(0.9)_brightness(0.62)]" aria-hidden="true" />;
}
