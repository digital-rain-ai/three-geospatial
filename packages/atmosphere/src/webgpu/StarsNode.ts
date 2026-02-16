import {
  HalfFloatType,
  Vector2,
  Matrix4,
} from 'three'
import {
  screenUV,
} from 'three/tsl'
import {
  NodeUpdateType,
  RendererUtils,
  RenderTarget,
  TempNode,
  type NodeBuilder,
  type NodeFrame,
  type TextureNode,
  type UniformNode
} from 'three/webgpu'

import { outputTexture } from '@takram/three-geospatial/webgpu'

import { DEFAULT_STARS_DATA_URL } from '../constants'
import { getAtmosphereContext } from './AtmosphereContext'
import { Stars } from './Stars'

const { resetRendererState, restoreRendererState } = RendererUtils

function createRenderTarget(): RenderTarget {
  const renderTarget = new RenderTarget(1, 1, {
    depthBuffer: false,
    type: HalfFloatType
  })
  const texture = renderTarget.texture
  texture.name = 'Stars'
  return renderTarget
}

const sizeScratch = /*#__PURE__*/ new Vector2()

export class StarsNode extends TempNode {
  static override get type(): string {
    return 'StarsNode'
  }

  stars: Stars

  private readonly textureNode: TextureNode
  private readonly renderTarget: RenderTarget
  private rendererState?: RendererUtils.RendererState

  private readonly previousCameraMatrixWorld = new Matrix4()
  private readonly previousProjectionMatrix = new Matrix4()
  private readonly previousMatrixECIToECEF = new Matrix4()
  private readonly previousMatrixECEFToWorld = new Matrix4()
  private previousPointSize = NaN
  private previousIntensity = NaN
  private previousWidth = -1
  private previousHeight = -1
  private hasPreviousState = false
  private needsRender = true

  constructor(data: string | ArrayBufferLike = DEFAULT_STARS_DATA_URL) {
    super('vec3')
    this.updateBeforeType = NodeUpdateType.FRAME

    this.stars = new Stars(data)
    this.renderTarget = createRenderTarget()

    this.textureNode = outputTexture(this, this.renderTarget.texture)
  }

  getTextureNode(): TextureNode {
    return this.textureNode
  }

  setSize(width: number, height: number): this {
    if (this.renderTarget.width !== width || this.renderTarget.height !== height) {
      this.renderTarget.setSize(width, height)
      this.needsRender = true
    }
    return this
  }

  private updateRenderState(
    camera: NonNullable<NodeFrame['camera']>,
    width: number,
    height: number
  ): boolean {
    const { matrixECIToECEF, matrixECEFToWorld } = this.atmosphereContext

    if (!this.hasPreviousState) {
      this.previousCameraMatrixWorld.copy(camera.matrixWorld)
      this.previousProjectionMatrix.copy(camera.projectionMatrix)
      this.previousMatrixECIToECEF.copy(matrixECIToECEF.value)
      this.previousMatrixECEFToWorld.copy(matrixECEFToWorld.value)
      this.previousPointSize = this.pointSize.value
      this.previousIntensity = this.intensity.value
      this.previousWidth = width
      this.previousHeight = height
      this.hasPreviousState = true
      return true
    }

    const changed =
      this.needsRender ||
      this.previousWidth !== width ||
      this.previousHeight !== height ||
      !this.previousCameraMatrixWorld.equals(camera.matrixWorld) ||
      !this.previousProjectionMatrix.equals(camera.projectionMatrix) ||
      !this.previousMatrixECIToECEF.equals(matrixECIToECEF.value) ||
      !this.previousMatrixECEFToWorld.equals(matrixECEFToWorld.value) ||
      this.previousPointSize !== this.pointSize.value ||
      this.previousIntensity !== this.intensity.value

    if (changed) {
      this.previousCameraMatrixWorld.copy(camera.matrixWorld)
      this.previousProjectionMatrix.copy(camera.projectionMatrix)
      this.previousMatrixECIToECEF.copy(matrixECIToECEF.value)
      this.previousMatrixECEFToWorld.copy(matrixECEFToWorld.value)
      this.previousPointSize = this.pointSize.value
      this.previousIntensity = this.intensity.value
      this.previousWidth = width
      this.previousHeight = height
      this.needsRender = false
    }

    return changed
  }

  override updateBefore(frame: NodeFrame): void {
    const { renderer } = frame
    const camera = this.stars.camera
    if (renderer == null || camera == null) {
      return
    }

    const size = renderer.getDrawingBufferSize(sizeScratch)
    this.setSize(size.x, size.y)

    if (!this.updateRenderState(camera, size.x, size.y)) {
      return
    }

    this.rendererState = resetRendererState(renderer, this.rendererState)

    renderer.setRenderTarget(this.renderTarget)
    renderer.render(this.stars, camera)

    restoreRendererState(renderer, this.rendererState)
  }

  override setup(builder: NodeBuilder): unknown {
    const atmosphereContext = getAtmosphereContext(builder)
    this.stars.camera = atmosphereContext.camera

    this.textureNode.uvNode = screenUV
    return this.textureNode
  }

  get pointSize(): UniformNode<number> {
    return this.stars.material.pointSize
  }

  set pointSize(value: UniformNode<number>) {
    this.stars.material.pointSize = value
  }

  get intensity(): UniformNode<number> {
    return this.stars.material.intensity
  }

  set intensity(value: UniformNode<number>) {
    this.stars.material.intensity = value
  }

  override dispose(): void {
    this.renderTarget.dispose()
    this.stars.dispose()
    super.dispose()
  }
}
