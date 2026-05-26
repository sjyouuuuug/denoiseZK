export function formatFixed(value: number, divisor: number, digits = 3): string {
  const scaled = value / divisor;
  if (Object.is(scaled, -0)) return "0";
  return Number(scaled.toFixed(digits)).toString();
}

export function formatFixedVector(values: number[], divisor: number, digits = 3): string {
  return values.map((value) => formatFixed(value, divisor, digits)).join(", ");
}
