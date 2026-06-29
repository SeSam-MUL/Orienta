/**
 * CrystalBackground — Canvas-based floating hexagonal particle network.
 * Renders behind Dashboard content. ~2% CPU, auto-pauses when tab hidden.
 */
import { useRef, useEffect } from 'react';

export default function CrystalBackground() {
  const canvasRef = useRef(null);
  const animRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const parent = canvas.parentElement;

    let prevW = 0, prevH = 0;
    function resize() {
      const w = parent.clientWidth;
      const h = parent.clientHeight;
      if (w === 0 || h === 0) return; // not visible yet
      canvas.width = w;
      canvas.height = h;
      // Redistribute particles when canvas gets its first real size or changes significantly
      if (prevW === 0 && w > 0) {
        for (const p of particles) {
          p.x = Math.random() * w;
          p.y = Math.random() * h;
        }
      }
      prevW = w;
      prevH = h;
    }

    const COUNT = 40;
    const particles = Array.from({ length: COUNT }, () => ({
      x: 0,
      y: 0,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      size: 2 + Math.random() * 3,
      rotation: Math.random() * Math.PI * 2,
      rotSpeed: (Math.random() - 0.5) * 0.004,
    }));

    resize();
    window.addEventListener('resize', resize);

    // Also observe container resize (parent may change size without window resize)
    const ro = new ResizeObserver(resize);
    ro.observe(parent);

    function drawHexagon(x, y, size, rotation) {
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const angle = rotation + (Math.PI / 3) * i;
        const px = x + size * Math.cos(angle);
        const py = y + size * Math.sin(angle);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.closePath();
    }

    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            ctx.strokeStyle = `rgba(189, 147, 249, ${(1 - dist / 120) * 0.1})`;
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }

      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        p.rotation += p.rotSpeed;
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        ctx.strokeStyle = 'rgba(189, 147, 249, 0.2)';
        ctx.lineWidth = 0.8;
        drawHexagon(p.x, p.y, p.size, p.rotation);
        ctx.stroke();

        ctx.fillStyle = 'rgba(189, 147, 249, 0.05)';
        drawHexagon(p.x, p.y, p.size, p.rotation);
        ctx.fill();
      }

      animRef.current = requestAnimationFrame(animate);
    }

    function onVisChange() {
      if (document.hidden) {
        cancelAnimationFrame(animRef.current);
      } else {
        animRef.current = requestAnimationFrame(animate);
      }
    }
    document.addEventListener('visibilitychange', onVisChange);
    animRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animRef.current);
      ro.disconnect();
      window.removeEventListener('resize', resize);
      document.removeEventListener('visibilitychange', onVisChange);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'absolute', inset: 0, zIndex: 0, pointerEvents: 'none' }}
    />
  );
}
