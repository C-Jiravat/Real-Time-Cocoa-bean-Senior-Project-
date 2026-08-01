import axios from 'axios'

const client = axios.create({ baseURL: '/api', withCredentials: true })

function message(error) {
  const response = error?.response
  if (!response) return 'ไม่สามารถเชื่อมต่อกับระบบได้ กรุณาลองใหม่อีกครั้ง'
  const detail = response.data?.detail ?? response.data
  if (typeof detail === 'string' && detail.trim()) return detail
  if (Array.isArray(detail)) return detail.map(item => item.msg || JSON.stringify(item)).join(' | ')
  return `คำขอไม่สำเร็จ (HTTP ${response.status})`
}

export async function login(email, password) {
  try { return (await client.post('/auth/login', { email, password })).data } catch (error) { throw new Error(message(error)) }
}
export async function logout() { return (await client.post('/auth/logout')).data }
export async function currentUser() {
  try { return (await client.get('/auth/me')).data } catch { return null }
}
export async function analyze(file, settings) {
  const form = new FormData()
  form.append('file', file)
  Object.entries(settings).forEach(([key, value]) => form.append(key, value))
  try { return (await client.post('/analysis', form)).data } catch (error) { throw new Error(message(error)) }
}
export async function benchmark(mode, files, settings) {
  const form = new FormData()
  form.append('target', files.target)
  if (mode === 'single') {
    form.append('image', files.image)
    if (files.target === 'both') {
      form.append('color_label', files.colorLabel)
      form.append('defect_label', files.defectLabel)
    } else form.append('label', files.label)
  } else form.append('archive', files.archive)
  Object.entries(settings).forEach(([key, value]) => form.append(key, value))
  try { return (await client.post(`/benchmark/${mode}`, form)).data } catch (error) { throw new Error(message(error)) }
}
